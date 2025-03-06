import os

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from utils import load_audio, load_optimus_ths_lib, speech_adjust, volumn_adjust


class StepAudio:
    def __init__(self, tokenizer_path: str, tts_path: str, llm_path: str, load_in_8bit: bool=False):
        load_optimus_ths_lib(os.path.join(llm_path, 'lib'))
        q_config = None
        if load_in_8bit:
            q_config = BitsAndBytesConfig(load_in_8bit=True)
            print(f"load in 8bit")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True, quantization_config=q_config,
        )
        self.encoder = StepAudioTokenizer(tokenizer_path)
        self.decoder = StepAudioTTS(tts_path, self.encoder)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=q_config,
        )

    def __call__(
        self,
        messages: list,
        speaker_id: str,
        speed_ratio: float = 1.0,
        volumn_ratio: float = 1.0,
    ):
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(text_with_audio, return_tensors="pt")
        outputs = self.llm.generate(
            token_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1] : -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        output_audio, sr = self.decoder(output_text, speaker_id)
        if speed_ratio != 1.0:
            output_audio = speech_adjust(output_audio, sr, speed_ratio)
        if volumn_ratio != 1.0:
            output_audio = volumn_adjust(output_audio, volumn_ratio)
        return output_text, output_audio, sr

    def encode_audio(self, audio_path):
        audio_wav, sr = load_audio(audio_path)
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                role = "human"
            if isinstance(content, str):
                text_with_audio += f"<|BOT|>{role}\n{content}<|EOT|>"
            elif isinstance(content, dict):
                if content["type"] == "text":
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content["type"] == "audio":
                    audio_tokens = self.encode_audio(content["audio"])
                    text_with_audio += f"<|BOT|>{role}\n{audio_tokens}<|EOT|>"
            elif content is None:
                text_with_audio += f"<|BOT|>{role}\n"
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        if not text_with_audio.endswith("<|BOT|>assistant\n"):
            text_with_audio += "<|BOT|>assistant\n"
        return text_with_audio

