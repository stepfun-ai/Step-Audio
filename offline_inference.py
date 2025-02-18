import torchaudio
import argparse
from stepaudio import StepAudio


def main():
    parser = argparse.ArgumentParser(description="StepAudio Offline Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Base path for model files"
    )
    args = parser.parse_args()

    model = StepAudio(
        tokenizer_path=f"{args.model_path}/Step-Audio-Tokenizer",
        tts_path=f"{args.model_path}/Step-Audio-TTS-3B",
        llm_path=f"{args.model_path}/Step-Audio-Chat",
    )

    # example for text input
    text, audio, sr = model(
        [{"role": "user", "content": "你好，我是你的朋友，我叫小明，你叫什么名字？"}],
        "Tingting",
    )
    torchaudio.save("output/output_e2e_tqta.wav", audio, sr)

    # example for audio input
    text, audio, sr = model(
        [
            {
                "role": "user",
                "content": {"type": "audio", "audio": "output/output_e2e_tqta.wav"},
            }
        ],
        "Tingting",
    )
    torchaudio.save("output/output_e2e_aqta.wav", audio, sr)


if __name__ == "__main__":
    main()
