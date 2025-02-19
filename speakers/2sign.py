
# 将立体音频转为单通道音频
from pydub import AudioSegment

# 加载 WAV 文件
audio = AudioSegment.from_wav("nezhaRAP_prompt.wav")

# 转换为单声道
mono_audio = audio.set_channels(1)

# 保存为新的 WAV 文件
mono_audio.export("nezhaRAP_prompt.wav", format="wav")
