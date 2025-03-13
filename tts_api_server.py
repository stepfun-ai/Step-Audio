import argparse
from datetime import datetime
from io import BytesIO
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
import torch
import torchaudio

from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS


def create_app(model_path: str):
    app = FastAPI()

    encoder = StepAudioTokenizer(f"{model_path}/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(f"{model_path}/Step-Audio-TTS-3B", encoder)
    
    @app.get("/voices", response_model=List[str])
    async def list_voices():
        """
        获取当前系统中所有可用的音色名称列表。
        """
        # 从 tts_engine.speakers_info 中获取所有音色名称
        available_voices = list(tts_engine.speakers_info.keys())
        
        return available_voices
    
    @app.post("/register_voice")
    async def register_voice(
        speaker_name: str = Form(...),
        prompt_text: str = Form(...),
        prompt_wav: UploadFile = File(...),
    ):
        """
        注册新的音色到系统中。
        
        :param speaker_name: 为这个声音指定一个唯一的名字
        :param prompt_wav: 参考音频文件
        :param prompt_text: 参考音频的文本内容
        """
        content = await prompt_wav.read()
        
        # 创建临时缓冲区并加载音频
        temp_buffer = BytesIO(content)
        prompt_wav, prompt_wav_sr = torchaudio.load(temp_buffer)
        
        # 处理多通道音频
        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            
        # 重采样到需要的采样率
        prompt_wav_16k = torchaudio.transforms.Resample(
            orig_freq=prompt_wav_sr, new_freq=16000
        )(prompt_wav)
        prompt_wav_22k = torchaudio.transforms.Resample(
            orig_freq=prompt_wav_sr, new_freq=22050
        )(prompt_wav)
        
        # 提取特征
        speech_feat, speech_feat_len = tts_engine.common_cosy_model.frontend._extract_speech_feat(prompt_wav_22k)
        speech_embedding = tts_engine.common_cosy_model.frontend._extract_spk_embedding(prompt_wav_16k)
        
        # 获取 prompt code
        prompt_code, _, _ = encoder.wav2token(prompt_wav, prompt_wav_sr)
        prompt_token = torch.tensor([prompt_code], dtype=torch.long) - 65536
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long)
        
        # 注册到 speakers_info
        tts_engine.speakers_info[speaker_name] = {
            "prompt_text": prompt_text,
            "prompt_code": prompt_code,
            "cosy_speech_feat": speech_feat.to(torch.bfloat16),
            "cosy_speech_feat_len": speech_feat_len,
            "cosy_speech_embedding": speech_embedding.to(torch.bfloat16),
            "cosy_prompt_token": prompt_token,
            "cosy_prompt_token_len": prompt_token_len,
        }
        
        return {
            "status": "success",
            "message": f"Successfully registered voice: {speaker_name}",
            "speaker_name": speaker_name
        }
    
    @app.post("/tts")
    async def text_to_speech(
        tts_text: str = Form(...),
        speaker: str = Form(default="Tingting"),
    ):
        # Generate audio
        output_audio, sr = tts_engine(tts_text, speaker)

        # Convert to WAV format in memory
        buffer = BytesIO()
        torchaudio.save(buffer, output_audio, sr, format="wav")
        buffer.seek(0)

        # Return audio data directly
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="tts_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav"'
            },
        )

    @app.post("/clone")
    async def voice_clone(
        tts_text: str = Form(...),
        prompt_text: str = Form(...),
        prompt_wav: UploadFile = File(...),
        speaker_name: str = Form(...),
    ):
        """
        使用提供的音频克隆声音并生成目标文本。
        
        :param speaker_name: 为这个声音指定一个唯一的名字（仅作为返回的文件名使用，不会注册）
        :param prompt_wav: 参考音频文件
        :param prompt_text: 参考音频的文本内容
        :param tts_text: 想要生成的文本内容
        """
        content = await prompt_wav.read()
        
        # 创建临时缓冲区
        temp_buffer = BytesIO(content)
        
        # 构建克隆说话人信息
        clone_speaker = {
            "wav_path": temp_buffer,
            "speaker": speaker_name,
            "prompt_text": prompt_text,
        }
        
        # 直接执行一次性克隆
        output_audio, sr = tts_engine(tts_text, "", clone_speaker)

        buffer = BytesIO()
        torchaudio.save(buffer, output_audio, sr, format="wav")
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="clone_{speaker_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.wav"',
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="StepAudio TTS Server")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Base path for model files"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
