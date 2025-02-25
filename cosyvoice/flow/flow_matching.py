# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import torch
import torch.nn.functional as F
from cosyvoice.matcha.flow_matching import BASECFM


class ConditionalCFM(BASECFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator
        self.inference_graphs = {}
        self.inference_buffers = {}
        # self.capture_inference()

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        prompt_len=0,
        # flow_cache=torch.zeros(1, 80, 0, 2),
        flow_cache=None,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature

        if flow_cache is not None:
            cache_size = flow_cache.shape[2]
            # fix prompt and overlap part mu and z
            if cache_size != 0:
                z[:, :, :cache_size] = flow_cache[:, :, :, 0]
                mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
            z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
            mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
            flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        ), flow_cache

    @torch.inference_mode()
    def capture_inference(self, seq_len_to_capture=list(range(128, 512, 8))):
        start_time = time.time()
        print(
            f"capture_inference for ConditionalCFM solve euler, seq_len_to_capture: {seq_len_to_capture}"
        )
        for seq_len in seq_len_to_capture:
            static_z = torch.randn(
                1, 80, seq_len, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            static_t_span = torch.linspace(
                0, 1, 11, device=torch.device("cuda"), dtype=torch.bfloat16
            )  # only capture at 10 steps
            static_mu = torch.randn(
                1, 80, seq_len, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            static_mask = torch.ones(
                1, 1, seq_len, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            static_spks = torch.randn(1, 80, device=torch.device("cuda"), dtype=torch.bfloat16)
            static_cond = torch.randn(
                1, 80, seq_len, device=torch.device("cuda"), dtype=torch.float32
            )
            static_out = torch.randn(
                1, 80, seq_len, device=torch.device("cuda"), dtype=torch.bfloat16
            )

            self._solve_euler_impl(
                static_z,
                t_span=static_t_span,
                mu=static_mu,
                mask=static_mask,
                spks=static_spks,
                cond=static_cond,
            )
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = self._solve_euler_impl(
                    static_z,
                    t_span=static_t_span,
                    mu=static_mu,
                    mask=static_mask,
                    spks=static_spks,
                    cond=static_cond,
                )

        self.inference_buffers[seq_len] = {
            "z": static_z,
            "t_span": static_t_span,
            "mu": static_mu,
            "mask": static_mask,
            "spks": static_spks,
            "cond": static_cond,
            "out": static_out,
        }
        self.inference_graphs[seq_len] = g
        end_time = time.time()
        print(
            f"capture_inference for ConditionalCFM solve euler, time elapsed: {end_time - start_time}"
        )

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        if hasattr(self, "inference_graphs") and len(self.inference_graphs) > 0:
            curr_seq_len = x.shape[2]

            available_lengths = sorted(list(self.inference_graphs.keys()))

            if curr_seq_len <= max(available_lengths):
                target_len = min(available_lengths, key=lambda x: abs(x - curr_seq_len))
                if target_len == curr_seq_len:
                    padded_x = x
                    padded_mu = mu
                    padded_mask = mask
                    if cond is not None:
                        padded_cond = cond
                else:
                    padded_x = torch.randn(
                        (x.shape[0], x.shape[1], target_len),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    padded_x[:, :, :curr_seq_len] = x

                    padded_mu = torch.randn(
                        (mu.shape[0], mu.shape[1], target_len),
                        dtype=mu.dtype,
                        device=mu.device,
                    )
                    padded_mu[:, :, :curr_seq_len] = mu

                    # FIXME(ys): uses zeros and maskgroupnorm
                    padded_mask = torch.ones(
                        (mask.shape[0], mask.shape[1], target_len),
                        dtype=mask.dtype,
                        device=mask.device,
                    )

                    if cond is not None:
                        padded_cond = torch.randn(
                            (cond.shape[0], cond.shape[1], target_len),
                            dtype=cond.dtype,
                            device=cond.device,
                        )
                        padded_cond[:, :, :curr_seq_len] = cond

                buffer = self.inference_buffers[target_len]
                buffer["z"].copy_(padded_x)
                buffer["t_span"].copy_(t_span)
                buffer["mu"].copy_(padded_mu)
                buffer["mask"].copy_(padded_mask)
                buffer["spks"].copy_(spks)
                if cond is not None:
                    buffer["cond"].copy_(padded_cond)

                self.inference_graphs[target_len].replay()

                output = buffer["out"][:, :, :curr_seq_len]
                return output

        return self._solve_euler_impl(x, t_span, mu, mask, spks, cond)

    def _solve_euler_impl(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            if self.inference_cfg_rate > 0:
                x_double = torch.cat([x, x], dim=0)
                mask_double = torch.cat([mask, mask], dim=0)
                mu_double = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                t_double = torch.cat([t, t], dim=0)
                spks_double = (
                    torch.cat([spks, torch.zeros_like(spks)], dim=0) if spks is not None else None
                )
                cond_double = torch.cat([cond, torch.zeros_like(cond)], dim=0)

                dphi_dt_double = self.forward_estimator(
                    x_double, mask_double, mu_double, t_double, spks_double, cond_double
                )

                dphi_dt, cfg_dphi_dt = torch.chunk(dphi_dt_double, 2, dim=0)
                dphi_dt = (
                    1.0 + self.inference_cfg_rate
                ) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.forward_estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        else:
            ort_inputs = {
                "x": x.cpu().numpy(),
                "mask": mask.cpu().numpy(),
                "mu": mu.cpu().numpy(),
                "t": t.cpu().numpy(),
                "spks": spks.cpu().numpy(),
                "cond": cond.cpu().numpy(),
            }
            output = self.estimator.run(None, ort_inputs)[0]
            return torch.tensor(output, dtype=x.dtype, device=x.device)

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
