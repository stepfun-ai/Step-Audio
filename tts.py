import math
import os
import re
import json
from threading import Thread, Lock
import uuid


import torchaudio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from cosyvoice.cli.cosyvoice import CosyVoice
from streamer import TokenStreamer


class RepetitionAwareLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1

        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size

        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


class StepAudioTTS:
    def __init__(
        self,
        model_path,
        encoder,
        device_map: str | dict | None = None,
        stream_factor: int = 2,
        **kwargs,
    ):
        # fast path to check params
        assert (
            stream_factor >= 2
        ), "stream_factor must >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)"
        self.stream_factor = stream_factor  # >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)

        # session ctx dict with lock, maybe need a session class
        self.session_lm_generat_lock = Lock()
        self.session_lm_generated_ids = {}  # session_id: ids(ptr)

        # load optimus_ths for flash attention, make sure LD_LIBRARY_PATH has `nvidia/cuda_nvrtc/lib`
        # if not, please manually set LD_LIBRARY_PATH=xxx/python3.10/site-packages/nvidia/cuda_nvrtc/lib
        try:
            if torch.__version__ >= "2.5":
                torch.ops.load_library(os.path.join(model_path, 'lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so'))
            elif torch.__version__ >= "2.3":
                torch.ops.load_library(os.path.join(model_path, 'lib/liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so'))
            elif torch.__version__ >= "2.2":
                torch.ops.load_library(os.path.join(model_path, 'lib/liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so'))
            print("Load optimus_ths successfully and flash attn would be enabled")
        except Exception as err:
            print(f"Fail to load optimus_ths and flash attn is disabled: {err}")

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if not device_map else device_map,
            trust_remote_code=True,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.common_cosy_model = CosyVoice(os.path.join(model_path, "CosyVoice-300M-25Hz"))
        self.music_cosy_model = CosyVoice(os.path.join(model_path, "CosyVoice-300M-25Hz-Music"))
        self.encoder = encoder
        self.sys_prompt_dict = {
            "sys_prompt_for_rap": "请参考对话历史里的音色，用RAP方式将文本内容大声说唱出来。",
            "sys_prompt_for_vocal": "请参考对话历史里的音色，用哼唱的方式将文本内容大声唱出来。",
            "sys_prompt_wo_spk": '作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n- "广东话"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n- "快速1"\n- "快速2"\n\n请在朗读时，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。',
            "sys_prompt_with_spk": '作为一名卓越的声优演员，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调来朗读文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n- "广东话"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n- "快速1"\n- "快速2"\n\n请在朗读时，使用[{}]的声音，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容自由演绎。',
        }
        self.register_speakers()

    def __call__(self, text: str, prompt_speaker: str, clone_dict: dict | None = None):
        prompt_speaker, prompt_speaker_info, cosy_model = self.preprocess_prompt(
            text, prompt_speaker, clone_dict=clone_dict
        )

        token_ids = self.tokenize(
            text,
            prompt_speaker_info["prompt_text"],
            prompt_speaker,
            prompt_speaker_info["prompt_code"],
        )
        output_ids = self.llm.generate(
            torch.tensor([token_ids]).to(torch.long).to("cuda"),
            max_length=8192,
            temperature=0.7,
            do_sample=True,
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
        )
        output_ids = output_ids[:, len(token_ids) : -1]  # skip eos token
        return (
            cosy_model.token_to_wav_offline(
                output_ids - 65536,
                prompt_speaker_info["cosy_speech_feat"].to(torch.bfloat16),
                prompt_speaker_info["cosy_speech_feat_len"],
                prompt_speaker_info["cosy_prompt_token"],
                prompt_speaker_info["cosy_prompt_token_len"],
                prompt_speaker_info["cosy_speech_embedding"].to(torch.bfloat16),
            ),
            22050,
        )

    def register_speakers(self):
        self.speakers_info = {}

        with open("speakers/speakers_info.json", "r") as f:
            speakers_info = json.load(f)

        for speaker_id, prompt_text in speakers_info.items():
            prompt_wav_path = f"speakers/{speaker_id}_prompt.wav"
            (
                prompt_code,
                prompt_token,
                prompt_token_len,
                speech_feat,
                speech_feat_len,
                speech_embedding,
            ) = self.preprocess_prompt_wav(prompt_wav_path)

            self.speakers_info[speaker_id] = {
                "prompt_text": prompt_text,
                "prompt_code": prompt_code,
                "cosy_speech_feat": speech_feat.to(torch.bfloat16),
                "cosy_speech_feat_len": speech_feat_len,
                "cosy_speech_embedding": speech_embedding.to(torch.bfloat16),
                "cosy_prompt_token": prompt_token,
                "cosy_prompt_token_len": prompt_token_len,
            }
            print(f"Registered speaker: {speaker_id}")

    def detect_instruction_name(self, text):
        instruction_name = ""
        match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
        if match_group is not None:
            instruction = match_group.group(1)
            instruction_name = instruction.strip("()（）")
        return instruction_name

    def tokenize(self, text: str, prompt_text: str, prompt_speaker: str, prompt_code: list):
        rap_or_vocal = self.detect_instruction_name(text) in ("RAP", "哼唱")

        if rap_or_vocal:
            if "哼唱" in text:
                prompt = self.sys_prompt_dict["sys_prompt_for_vocal"]
            else:
                prompt = self.sys_prompt_dict["sys_prompt_for_rap"]
        elif prompt_speaker:
            prompt = self.sys_prompt_dict["sys_prompt_with_spk"].format(prompt_speaker)
        else:
            prompt = self.sys_prompt_dict["sys_prompt_wo_spk"]

        sys_tokens = self.tokenizer.encode(f"system\n{prompt}")

        history = [1]
        history.extend([4] + sys_tokens + [3])

        _prefix_tokens = self.tokenizer.encode("\n")
        prompt_token_encode = self.tokenizer.encode("\n" + prompt_text)
        prompt_tokens = prompt_token_encode[len(_prefix_tokens) :]

        target_token_encode = self.tokenizer.encode("\n" + text)
        target_tokens = target_token_encode[len(_prefix_tokens) :]

        qrole_toks = self.tokenizer.encode("human\n")
        arole_toks = self.tokenizer.encode("assistant\n")

        history.extend(
            [4]
            + qrole_toks
            + prompt_tokens
            + [3]
            + [4]
            + arole_toks
            + prompt_code
            + [3]
            + [4]
            + qrole_toks
            + target_tokens
            + [3]
            + [4]
            + arole_toks
        )
        return history

    def preprocess_prompt_wav(self, prompt_wav_path: str):
        prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 将多通道音频转换为单通道
        prompt_wav_16k = torchaudio.transforms.Resample(orig_freq=prompt_wav_sr, new_freq=16000)(
            prompt_wav
        )
        prompt_wav_22k = torchaudio.transforms.Resample(orig_freq=prompt_wav_sr, new_freq=22050)(
            prompt_wav
        )

        speech_feat, speech_feat_len = self.common_cosy_model.frontend._extract_speech_feat(
            prompt_wav_22k
        )
        speech_embedding = self.common_cosy_model.frontend._extract_spk_embedding(prompt_wav_16k)

        prompt_code, _, _ = self.encoder.wav2token(prompt_wav, prompt_wav_sr)
        prompt_token = torch.tensor([prompt_code], dtype=torch.long) - 65536
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long)

        return (
            prompt_code,
            prompt_token,
            prompt_token_len,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )

    def preprocess_prompt(self, text: str, prompt_speaker: str, clone_dict: dict | None = None):
        if clone_dict:
            (
                clone_prompt_code,
                clone_prompt_token,
                clone_prompt_token_len,
                clone_speech_feat,
                clone_speech_feat_len,
                clone_speech_embedding,
            ) = self.preprocess_prompt_wav(clone_dict["wav_path"])
            prompt_speaker = clone_dict["speaker"]
            self.speakers_info[prompt_speaker] = {
                "prompt_text": clone_dict["prompt_text"],
                "prompt_code": clone_prompt_code,
                "cosy_speech_feat": clone_speech_feat.to(torch.bfloat16),
                "cosy_speech_feat_len": clone_speech_feat_len,
                "cosy_speech_embedding": clone_speech_embedding.to(torch.bfloat16),
                "cosy_prompt_token": clone_prompt_token,
                "cosy_prompt_token_len": clone_prompt_token_len,
            }

        instruction_name = self.detect_instruction_name(text)
        prompt_speaker_info = self.speakers_info[prompt_speaker]

        if instruction_name in ("RAP", "哼唱"):
            if not clone_dict:
                prompt_speaker_info = self.speakers_info[f"{prompt_speaker}{instruction_name}"]
            cosy_model = self.music_cosy_model
        else:
            cosy_model = self.common_cosy_model

        return prompt_speaker, prompt_speaker_info, cosy_model

    @torch.inference_mode()
    def static_batch_stream(
        self,
        text: str,
        prompt_speaker: str,
        clone_dict: dict | None = None,
        session_id: str = str(uuid.uuid4()),
    ):
        """
        - step1 lm stream generate token
        - static batch size to gen waveform
            - flow: audio vq tokens to mel
            - hifi: mel to waveform
        """
        prompt_speaker, prompt_speaker_info, cosy_model = self.preprocess_prompt(
            text, prompt_speaker, clone_dict=clone_dict
        )
        output_audio_sample_rate = cosy_model.model.hift.sampling_rate

        token_ids = self.tokenize(
            text,
            prompt_speaker_info["prompt_text"],
            prompt_speaker,
            prompt_speaker_info["prompt_code"],
        )

        # session streamer
        streamer = TokenStreamer(skip_prompt=True)

        generation_kwargs = dict(
            input_ids=torch.tensor([token_ids]).to(torch.long).to("cuda"),
            eos_token_id=3,
            streamer=streamer,
            max_length=8192,
            temperature=0.7,
            do_sample=True,
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
        )
        # print("generation_kwargs", generation_kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        with self.session_lm_generat_lock:
            self.session_lm_generated_ids[session_id] = []

        batch_size = math.ceil(self.stream_factor * cosy_model.model.flow.input_frame_rate)
        for token_id in streamer:
            # print(token_id, end=",", flush=True)
            if token_id == 3:  # skip <|EOT|>, break
                break
            self.session_lm_generated_ids[session_id].append(token_id)
            if len(self.session_lm_generated_ids[session_id]) % batch_size == 0:
                batch = (
                    torch.tensor(self.session_lm_generated_ids[session_id])
                    .unsqueeze(0)
                    .to(cosy_model.model.device)
                    - 65536
                )  # [T] -> [1,T]
                # Process each batch
                sub_tts_speech = cosy_model.token_to_wav_offline(
                    batch,
                    prompt_speaker_info["cosy_speech_feat"].to(torch.bfloat16),
                    prompt_speaker_info["cosy_speech_feat_len"],
                    prompt_speaker_info["cosy_prompt_token"],
                    prompt_speaker_info["cosy_prompt_token_len"],
                    prompt_speaker_info["cosy_speech_embedding"].to(torch.bfloat16),
                )
                yield {"tts_speech": sub_tts_speech, "sample_rate": output_audio_sample_rate}
                with self.session_lm_generat_lock:
                    self.session_lm_generated_ids[session_id] = []

        if len(self.session_lm_generated_ids[session_id]) > 0:
            batch = (
                torch.tensor(self.session_lm_generated_ids[session_id])
                .unsqueeze(0)
                .to(cosy_model.model.device)
                - 65536
            )  # [T] -> [1,T]
            # Process each batch
            sub_tts_speech = cosy_model.token_to_wav_offline(
                batch,
                prompt_speaker_info["cosy_speech_feat"].to(torch.bfloat16),
                prompt_speaker_info["cosy_speech_feat_len"],
                prompt_speaker_info["cosy_prompt_token"],
                prompt_speaker_info["cosy_prompt_token_len"],
                prompt_speaker_info["cosy_speech_embedding"].to(torch.bfloat16),
            )
            yield {"tts_speech": sub_tts_speech, "sample_rate": output_audio_sample_rate}

        with self.session_lm_generat_lock:
            self.session_lm_generated_ids.pop(session_id)
        torch.cuda.empty_cache()
