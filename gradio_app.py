import gradio as gr
# import torch
import torchaudio
import argparse
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from utils import load_audio
import os
from datetime import datetime

# 修改模型地址 
model_path = "/models"
encoder = StepAudioTokenizer(f"{model_path}/Step-Audio-Tokenizer")
tts_engine = StepAudioTTS(f"{model_path}/Step-Audio-TTS-3B", encoder)

# 假设 StepAudioTTS 已经初始化
# step_audio_tts = StepAudioTTS()

# 处理普通语音合成


# 普通语音合成
def tts_common(text,speaker, emotion, language, speed):
    text = f"({emotion})({language})({speed})" + text
    print(text)
    # speaker = "Tingting"
    # speaker = "xueqin"
    output_audio, sr = tts_engine(text, speaker)
    # 获取当前时间并格式化文件名
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"./results/common/{current_time}.wav"

    # 保存音频文件
    torchaudio.save(save_path, output_audio, sr)

    return save_path  # 返回音频文件路径
def tts_music(text_input_rap,speaker, mode_input):
    text_input_rap = f"({mode_input})" + text_input_rap
    print(text_input_rap)
    # speaker = "Tingting"
    output_audio, sr = tts_engine(text_input_rap, speaker)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"./results/music/{current_time}.wav"

    # 保存音频文件
    torchaudio.save(save_path, output_audio, sr)

    return save_path  # 返回音频文件路径

# 处理语音克隆
def tts_clone(text, wav_file,speaker_prompt, speaker_name, emotion, language, speed):
    clone_speaker = {
        "wav_path": wav_file,
        "speaker": speaker_name,
        "prompt_text": speaker_prompt
    }
    clone_text = f"({emotion})({language})({speed})" + text
    print(clone_text)
    output_audio, sr = tts_engine(clone_text, "",clone_speaker)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"./results/clone/{current_time}.wav"

    # 保存音频文件
    torchaudio.save(save_path, output_audio, sr)

    return save_path  # 返回音频文件路径

# 选项列表
emotion_options = ["高兴1", "高兴2", "生气1", "生气2", "悲伤1", "撒娇1"]
language_options = ["中文", "英文", "韩语", "日语", "四川话", "粤语", "广东话"]
speed_options = ["慢速1", "慢速2", "快速1", "快速2"]
speaker_options = ["Tingting","nezha"]

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Step-Audio-TTS-3B Demo")
    gr.Markdown(
        """
    <p align="center">
        <img src="assets/logo.png"  height=100>
    </p>
        """)

    with gr.Tab("普通语音合成"):
        #四个输入参数
        text_input = gr.Textbox(label="输入文本")

        # 允许用户不选择或者输入自定义值
        speaker_input = gr.Dropdown(speaker_options, label="选择讲话人（默认Tingting）")
        emotion_input = gr.Dropdown(emotion_options, label="选择情感（可选）", allow_custom_value=True, interactive=True)
        language_input = gr.Dropdown(language_options, label="选择语种/方言（可选）", allow_custom_value=True, interactive=True)
        speed_input = gr.Dropdown(speed_options, label="语速调整（可选）", allow_custom_value=True, interactive=True)

        # 点击生成
        submit_btn = gr.Button("🔊 生成语音")
        #生成的语音
        output_audio = gr.Audio(label="合成语音", interactive=False)

        submit_btn.click(tts_common, inputs=[text_input, speaker_input,emotion_input, language_input, speed_input], outputs=output_audio)



    with gr.Tab("RAP / 哼唱模式"):
        text_input_rap = gr.Textbox(label="输入文本")
        speaker_input = gr.Dropdown(speaker_options, label="选择讲话人（默认Tingting）")
        mode_input = gr.Radio(["RAP", "哼唱"], label="模式选择")
        submit_btn_rap = gr.Button("🎤 生成 RAP / 哼唱")
        output_audio_rap = gr.Audio(label="合成语音", interactive=False)
        submit_btn_rap.click(tts_music, inputs=[text_input_rap, speaker_input,mode_input], outputs=output_audio_rap)

    with gr.Tab("语音克隆"):
        text_input_clone = gr.Textbox(label="输入文本")
        audio_input = gr.File(label="上传参考音频 (.wav)")
        speaker_prompt = gr.Textbox(label="音频prompt", placeholder="音频的文本内容")
        speaker_name_input = gr.Textbox(label="为克隆声音命名", placeholder="如：MyVoice")

        # 允许用户不选择或者输入自定义值
        emotion_input = gr.Dropdown(emotion_options, label="选择情感（可选）", allow_custom_value=True, interactive=True)
        language_input = gr.Dropdown(language_options, label="选择语种/方言（可选）", allow_custom_value=True, interactive=True)
        speed_input = gr.Dropdown(speed_options, label="语速调整（可选）", allow_custom_value=True, interactive=True)

        submit_btn_clone = gr.Button("🗣️ 生成克隆语音")
        output_audio_clone = gr.Audio(label="合成语音", interactive=False)
        

        submit_btn_clone.click(tts_clone, 
                               inputs=[text_input_clone, audio_input, speaker_prompt,speaker_name_input, emotion_input, language_input, speed_input], 
                               outputs=output_audio_clone)

demo.launch(server_name = "0.0.0.0",server_port=8080)
