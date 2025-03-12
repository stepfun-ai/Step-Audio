import gradio as gr
import argparse
import torchaudio
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from datetime import datetime
import os


# 保存音频
def save_audio(audio_type, audio_data, sr):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchaudio.save(save_path, audio_data, sr)
    return save_path


# 普通语音合成
def tts_common(text, speaker, emotion, language, speed):
    text = (
        (f"({emotion})" if emotion else "")
        + (f"({language})" if language else "")
        + (f"({speed})" if speed else "")
        + text
    )
    output_audio, sr = tts_engine(text, speaker)
    audio_type = "common"
    common_path = save_audio(audio_type, output_audio, sr)
    return common_path


# RAP / 哼唱模式
def tts_music(text_input_rap, speaker, mode_input):
    text_input_rap = f"({mode_input})" + text_input_rap
    output_audio, sr = tts_engine(text_input_rap, speaker)
    audio_type = "music"
    music_path = save_audio(audio_type, output_audio, sr)
    return music_path


# 语音克隆
def tts_clone(text, wav_file, speaker_prompt, emotion, language, speed):
    clone_speaker = {
        "wav_path": wav_file,
        "speaker": "custom_voice",
        "prompt_text": speaker_prompt,
    }
    clone_text = (
        (f"({emotion})" if emotion else "")
        + (f"({language})" if language else "")
        + (f"({speed})" if speed else "")
        + text
    )
    output_audio, sr = tts_engine(clone_text, "", clone_speaker)
    audio_type = "clone"
    clone_path = save_audio(audio_type, output_audio, sr)
    return clone_path


def launch_demo(args):
    # 选项列表
    emotion_options = ["高兴1", "高兴2", "生气1", "生气2", "悲伤1", "撒娇1"]
    language_options = ["中文", "英文", "韩语", "日语", "四川话", "粤语", "广东话"]
    speed_options = ["慢速1", "慢速2", "快速1", "快速2"]
    speaker_options = ["Tingting"]
    # Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("## 🎙️ Step-Audio-TTS-3B Demo")

        # 普通语音合成
        with gr.Tab("Common TTS (普通语音合成)"):
            text_input = gr.Textbox(
                label="Input Text (输入文本)",
            )
            speaker_input = gr.Dropdown(
                speaker_options,
                label="Speaker Selection (音色选择)",
            )
            emotion_input = gr.Dropdown(
                emotion_options,
                label="Emotion Style (情感风格)",
                allow_custom_value=True,
                interactive=True,
            )
            language_input = gr.Dropdown(
                language_options,
                label="Language/Dialect (语言/方言)",
                allow_custom_value=True,
                interactive=True,
            )
            speed_input = gr.Dropdown(
                speed_options,
                label="Speech Rate (语速调节)",
                allow_custom_value=True,
                interactive=True,
            )
            submit_btn = gr.Button("🔊 Generate Speech (生成语音)")
            output_audio = gr.Audio(
                label="Output Audio (合成语音)",
                interactive=False,
            )

            submit_btn.click(
                tts_common,
                inputs=[
                    text_input,
                    speaker_input,
                    emotion_input,
                    language_input,
                    speed_input,
                ],
                outputs=output_audio,
            )

        # RAP / 哼唱模式
        with gr.Tab("RAP/Humming Mode (RAP/哼唱模式)"):
            text_input_rap = gr.Textbox(
                label="Lyrics Input (歌词输入)",
            )
            speaker_input = gr.Dropdown(
                speaker_options,
                label="Speaker Selection (音色选择)",
            )
            mode_input = gr.Radio(
                ["RAP", "Humming (哼唱)"],
                value="RAP",
                label="Generation Mode (生成模式)",
            )
            submit_btn_rap = gr.Button("🎤 Generate Performance (生成演绎)")
            output_audio_rap = gr.Audio(
                label="Performance Audio (演绎音频)", interactive=False
            )
            submit_btn_rap.click(
                tts_music,
                inputs=[text_input_rap, speaker_input, mode_input],
                outputs=output_audio_rap,
            )

        with gr.Tab("Voice Clone (语音克隆)"):
            text_input_clone = gr.Textbox(
                label="Target Text (目标文本)",
                placeholder="Text to be synthesized with cloned voice (待克隆语音合成的文本)",
            )
            audio_input = gr.File(
                label="Reference Audio Upload (参考音频上传)",
            )
            speaker_prompt = gr.Textbox(
                label="Exact text from reference audio (输入参考音频的准确文本)",
            )
            emotion_input = gr.Dropdown(
                emotion_options,
                label="Emotion Style (情感风格)",
                allow_custom_value=True,
                interactive=True,
            )
            language_input = gr.Dropdown(
                language_options,
                label="Language/Dialect (语言/方言)",
                allow_custom_value=True,
                interactive=True,
            )
            speed_input = gr.Dropdown(
                speed_options,
                label="Speech Rate (语速调节)",
                allow_custom_value=True,
                interactive=True,
            )
            submit_btn_clone = gr.Button("🗣️ Synthesize Cloned Speech (合成克隆语音)")
            output_audio_clone = gr.Audio(
                label="Cloned Speech Output (克隆语音输出)",
                interactive=False,
            )
            submit_btn_clone.click(
                tts_clone,
                inputs=[
                    text_input_clone,
                    audio_input,
                    speaker_prompt,
                    emotion_input,
                    language_input,
                    speed_input,
                ],
                outputs=output_audio_clone,
            )

    # 启动 Gradio demo
    demo.queue().launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Demo server port."
    )
    parser.add_argument("--tmp_dir", type=str, default="/tmp/gradio", help="Save path.")

    args = parser.parse_args()
    # 使用解析后的命令行参数设置模型路径
    model_path = args.model_path
    encoder = StepAudioTokenizer(os.path.join(model_path, "Step-Audio-Tokenizer"))
    tts_engine = StepAudioTTS(os.path.join(model_path, "Step-Audio-TTS-3B"), encoder)
    launch_demo(args)
