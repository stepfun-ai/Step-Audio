import io
import base64

import librosa
import numpy as np
import math
import torch
import torchaudio
import torchaudio
import sox
import tempfile


def encode_wav(wav, sr, rep_format="wav"):
    with io.BytesIO() as wavio:
        torchaudio.save(wavio, wav, sr, format=rep_format)
        audio_bytes = wavio.getvalue()
        encoded_wav = base64.b64encode(audio_bytes).decode("ascii")
    return encoded_wav


def trim_silence(audio, sr, keep_left_time=0.05, keep_right_time=0.22, hop_size=240):
    _, index = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=128)
    num_frames = int(math.ceil((index[1] - index[0]) / hop_size))  # 300

    left_sil_samples = int(keep_left_time * sr)
    right_sil_samples = int(keep_right_time * sr)

    wav_len = len(audio)
    start_idx = index[0] - left_sil_samples
    trim_wav = audio

    if start_idx > 0:
        trim_wav = trim_wav[start_idx:]
    else:
        trim_wav = np.pad(trim_wav, (abs(start_idx), 0), mode="constant", constant_values=0.0)
    wav_len = len(trim_wav)
    out_len = int(num_frames * hop_size + (keep_left_time + keep_right_time) * sr)

    if out_len < wav_len:
        trim_wav = trim_wav[:out_len]
    else:
        trim_wav = np.pad(trim_wav, (0, (out_len - wav_len)), mode="constant", constant_values=0.0)
    return trim_wav


def volumn_adjust(audio16bit_torch, sr, volumn_ratio):
    """使用sox进行音频音量调整
    Args:
        audio16bit_torch (Tensor): 输入音频张量 [1, samples]
        volume_ratio (float): 音量比率，>1增大音量，<1降低音量

    Returns:
        Tensor: 调整音量后的音频张量
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=True
    ) as temp_in, tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_out:
        # 保存输入音频到临时文件
        torchaudio.save(temp_in.name, audio16bit_torch, sr)  # 假设采样率为16000
        # 创建sox转换器
        tfm = sox.Transformer()
        tfm.vol(volumn_ratio)  # 设置音量调整比率
        # 应用音量调整
        tfm.build_file(temp_in.name, temp_out.name)
        # 读取处理后的音频
        audio_changed, _ = torchaudio.load(temp_out.name)
    return audio_changed


def speech_adjust(audio16bit_torch, sr, speed_ratio):
    """使用sox进行音频变速处理
    Args:
        audio16bit_torch (Tensor): 输入音频张量 [1, samples]
        speed_ratio (float): 速度比率，>1加速，<1减速

    Returns:
        Tensor: 变速后的音频张量
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=True
    ) as temp_in, tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_out:
        # 保存输入音频到临时文件
        torchaudio.save(temp_in.name, audio16bit_torch, sr)  # 假设采样率为16000
        # 创建sox转换器
        tfm = sox.Transformer()
        tfm.tempo(speed_ratio)  # 设置变速比率
        # 应用变速处理
        tfm.build_file(temp_in.name, temp_out.name)
        # 读取处理后的音频
        audio_changed, _ = torchaudio.load(temp_out.name)
    return audio_changed


def audio_resample(audio16bit_torch, result_sr, target_sample_rate):
    audio16bit_torch = torchaudio.transforms.Resample(
        orig_freq=result_sr, new_freq=target_sample_rate
    )(audio16bit_torch)
    result_sr = target_sample_rate
    return audio16bit_torch, result_sr


def norm_audio(audio16bit_torch):
    # 直接 归一化处理。
    audio16bit_torch = audio16bit_torch.numpy()
    audio16bit_torch = (audio16bit_torch / np.abs(audio16bit_torch).max() * 32767).astype(np.int16)
    audio16bit_torch = torch.from_numpy(audio16bit_torch)
    return audio16bit_torch


def resample_audio(wav, original_sample_rate, target_sample_rate):
    if original_sample_rate != target_sample_rate:
        assert (
            original_sample_rate > target_sample_rate
        ), "wav sample rate {} must be greater than {}".format(
            original_sample_rate, target_sample_rate
        )
        wav = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, new_freq=target_sample_rate
        )(wav)
    return wav


def energy_norm_fn(wav):
    if type(wav) is np.ndarray:
        max_data = np.max(np.abs(wav))
        wav = wav / max(max_data, 0.01) * 0.999
    else:
        max_data = torch.max(torch.abs(wav))
        wav = wav / max(max_data, 0.01) * 0.999
    return wav


def get_audio_tokens(audio_tokens: str) -> list[int]:
    audio_tokens = audio_tokens.split("><audio_")
    audio_tokens = [
        int(token.replace("<audio_", "").replace(">", "")) + 65536 for token in audio_tokens
    ]
    return audio_tokens


def load_audio(audio_path: str):
    audio_wav, sr = torchaudio.load(audio_path)
    audio_wav = audio_wav.mean(dim=0, keepdim=True)
    return audio_wav, sr


def splite_batches(tensor_audio_token_ids: torch.Tensor, batch_size: int):
    """
    splite batches of audio token IDs.
    # Assuming tensor_audio_token_ids is already defined as in your original code
    # and has shape (1, sequence_length)

    Args:
      tensor_audio_token_ids: A tensor of audio token IDs.
      batch_size: The desired batch size.

    Returns:
      A list of tensors, where each tensor is a batch of audio token IDs.
    """
    sequence_length = tensor_audio_token_ids.shape[1]
    num_batches = (sequence_length + batch_size - 1) // batch_size
    batched_ids = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, sequence_length)
        batch = tensor_audio_token_ids[:, start_index:end_index]
        batched_ids.append(batch)

    return batched_ids


def merge_tensors(sub_tts_speechs: torch.Tensor):
    """
    Merges a list of tensors into a single tensor.
    # Assuming sub_tts_speechs is a list of tensors
    # and all tensors in the list have the same number of channels
    # and has shape (1, sequence_length)
    # but possibly different lengths.

    Args:
      sub_tts_speechs: A list of tensors.

    Returns:
      A single tensor with all the sub tensors concatenated along the time dimension.
      Returns None if the input list is empty or contains tensors with inconsistent shapes.
    """
    if not sub_tts_speechs:
        return None

    num_channels = sub_tts_speechs[0].shape[0]
    total_length = sum(tensor.shape[1] for tensor in sub_tts_speechs)
    merged_tensor = torch.empty(
        num_channels, total_length, dtype=sub_tts_speechs[0].dtype, device=sub_tts_speechs[0].device
    )
    current_position = 0

    for tensor in sub_tts_speechs:
        if tensor.shape[0] != num_channels:
            print("Error: Tensors have inconsistent number of channels.")
            return None

        merged_tensor[:, current_position : current_position + tensor.shape[1]] = tensor
        current_position += tensor.shape[1]

    return merged_tensor
