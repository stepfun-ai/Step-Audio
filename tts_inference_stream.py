import os
import argparse

import torchaudio

from tokenizer import StepAudioTokenizer
from utils import merge_tensors
from tts import StepAudioTTS


def main():
    parser = argparse.ArgumentParser(description="StepAudio Stream Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Base path for model files")
    parser.add_argument(
        "--synthesis-type", type=str, default="tts", help="Use tts or Clone for Synthesis"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for synthesis audios"
    )
    parser.add_argument(
        "--stream", type=str, default="static_batch", help="Synthesis audios with streaming"
    )
    parser.add_argument(
        "--stream-factor", type=int, default=2, help="Synthesis audios stream factor"
    )
    parser.add_argument(
        "--stream-scale-factor", type=float, default=1.0, help="Synthesis audios stream scale factor"
    )
    parser.add_argument(
        "--max-stream-factor", type=int, default=2, help="Synthesis audios max stream factor"
    )
    parser.add_argument(
        "--token-overlap-len", type=int, default=20, help="Synthesis audios token overlap len"
    )
    args = parser.parse_args()
    os.makedirs(f"{args.output_path}", exist_ok=True)

    encoder = StepAudioTokenizer(f"{args.model_path}/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(
        f"{args.model_path}/Step-Audio-TTS-3B",
        encoder,
        stream_factor=args.stream_factor,
        stream_scale_factor=args.stream_scale_factor,
        max_stream_factor=args.max_stream_factor,
        token_overlap_len=args.token_overlap_len,
    )

    if args.synthesis_type == "tts":
        text = "（RAP）君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        text = os.getenv("TTS_TEXT", text)
        batch_stream = tts_engine.batch_stream(text, "Tingting")
        sub_tts_speechs = []
        sr = 22050
        for item in batch_stream:
            sr = item["sample_rate"]
            sub_tts_speechs.append(item["tts_speech"])
        output_audio = merge_tensors(sub_tts_speechs)  # [1,T]
        torchaudio.save(f"{args.output_path}/output_tts_stream.wav", output_audio, sr)
    else:
        clone_speaker = {
            "speaker": "test",
            "prompt_text": "叫做秋风起蟹脚痒，啊，什么意思呢？就是说这秋风一起啊，螃蟹就该上市了。",
            "wav_path": "examples/prompt_wav_yuqian.wav",
        }
        text_clone = "万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        text_clone = os.getenv("TTS_TEXT", text_clone)
        batch_stream = tts_engine.batch_stream(text_clone, "", clone_speaker)
        sub_tts_speechs = []
        sr = 22050
        for item in batch_stream:
            sr = item["sample_rate"]
            sub_tts_speechs.append(item["tts_speech"])
        output_audio = merge_tensors(sub_tts_speechs)  # [1,T]
        torchaudio.save(f"{args.output_path}/output_clone_stream.wav", output_audio, sr)


if __name__ == "__main__":
    main()
