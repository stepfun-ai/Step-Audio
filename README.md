# Step-Audio

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Summary](#2-model-summary)
3. [Model Download](#3-model-download)
4. [Model Usage](#4-model-usage)
5. [Benchmark](#5-benchmark)
6. [Online Engine](#6-online-engine)
7. [Citation](#7-citation)

## 1. Introduction

Step-Audio is the first production-ready open-source framework for intelligent speech interaction that harmonizes comprehension and generation, supporting multilingual conversations (e.g., Chinese, English, Japanese), emotional tones (e.g., joy/sadness), regional dialects (e.g., Cantonese/Sichuanese), adjustable speech rates, and prosodic styles (e.g., rap). Step-Audio demonstrates four key technical innovations:

- **130B-Parameter Multimodal Model**: A single unified model integrating comprehension and generation capabilities, performing speech recognition, semantic understanding, dialogue, voice cloning, and speech synthesis. We have made the 130B Step-Audio-Chat variant open source.
    
- **Generative Data Engine**: Eliminates traditional TTS's reliance on manual data collection by generating high-quality audio through our 130B-parameter multimodal model. Leverages this data to train and publicly release a resource-efficient Step-Audio-TTS-3B model with enhanced instruction-following capabilities for controllable speech synthesis.
    
- **Granular Voice Control**: Enables precise regulation through instruction-based control design, supporting multiple emotions (anger, joy, sadness), dialects (Cantonese, Sichuanese, etc.), and vocal styles (rap, a cappella humming) to meet diverse speech generation needs.
    
- **Enhanced Intelligence**: Improves agent performance in complex tasks through ToolCall mechanism integration and role-playing enhancements.
\end{itemize}

## 2. Model Summary
In Step-Audio, audio streams are tokenized via a dual-codebook framework, combining parallel semantic (16.7Hz, 1024-entry codebook) and acoustic (25Hz, 4096-entry codebook) tokenizers with 2:3 temporal interleaving. A 130B-parameter LLM foundation (Step-1) is further enhanced via audio-contextualized continual pretraining and task-specific post-training, enabling robust cross-modal speech understanding. A hybrid speech decoder combining flow matching with neural vocoding, optimized for real-time waveform generation. A streaming-aware architecture featuring speculative response generation (40\% commit rate) and text-based context management (14:1 compression ratio) for efficient cross-modal alignment.
![Architecture](assets/architecture.png)

### 2.1 Tokenizer

We implement a token-level interleaving approach to effectively integrate semantic tokenization and acoustic tokenization. The semantic tokenizer employs a codebook size of 1024, while the acoustic tokenizer utilizes a larger codebook size of 4096 to capture finer acoustic details. Given the differing token rates, we establish a temporal alignment ratio of 2:3, where every two semantic tokens are paired with three acoustic tokens.

### 2.2 Language Model

To enhance Step-Audio’s ability to effectively process speech information and
achieve accurate speech-text alignment, we conducted audio continual pretrain-ing based on Step-1, a 130-billion parameter pretrained text-based large language model (LLM). 

### 2.3 Speech Decoder
The speech decoder in Step-Audio serves a critical function in converting discrete speech tokens, which contain both semantic and acoustic information, into continuous time-domain waveforms that represent natural speech. The decoder architecture incorporates a flow matching model and a mel-to-wave vocoder. To optimize the intelligibility and naturalness of the synthesized speech, the speech decoder is trained using a dual-code interleaving approach, ensuring seamless integration of semantic and acoustic features throughout the generation process.

### 2.4 Real-time Inference Pipeline
To enable real-time interactions, we have designed an optimized inference pipeline. At its core, the Controller module manages state transitions, orchestrates speculative response generation, and ensures seamless coordination between critical subsystems. These subsystems include Voice Activity Detection (VAD) for detecting user speech, the Streaming Audio Tokenizer for processing audio in real-time, the Step-Audio language model and Speech Decoder for processing and generating responses, and the Context Manager for preserving conversational continuity.
![Inference Pipeline](assets/pipeline.png)

### 2.5 Post training details
In the post-training phase, we conducted task-specific Supervised Fine-Tuning (SFT) for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS). For Audio Input Text Output (AQTA) tasks, we implemented SFT using diversified high-quality datasets combined with Reinforcement Learning from Human Feedback (RLHF) to enhance response quality, enabling fine-grained control over emotional expression, speech speed, dialect, and prosody.
![RLHF](assets/rlhf.png)


## 3. Model Download
| Models   | Links   |
|-------|-------|
| Step-Audio-Tokenizer | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-Chat | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B) |
| Step-Audio-TTS-3B | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-Chat) |

## 4. Model Usage
### 📜 4.1  Requirements
The following table shows the requirements for running Step-Audio model (batch size = 1):

|     Model    |  Setting<br/>(sample frequency) | GPU Minimum Memory  |
|------------|--------------------------------|----------------|
| Step-Audio-Tokenizer   |        41.6Hz          |       1.5GB        |
| Step-Audio-Chat   |        41.6Hz          |       265GB        |
| Step-Audio-TTS-3B   |        25Hz          |       8GB        |

* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a four A800 80G GPU.
  * **Recommended**: We recommend using 4xA800/H800 GPU with 80GB memory for better generation quality.
* Tested operating system: Linux

### 🔧 4.2 Dependencies and Installation
- Python >= 3.10.0 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.3-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
git clone https://github.com/stepfun-ai/Step-Audio.git
conda create -n stepaudio python=3.10
conda activate stepaudio

cd Step-Audio
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-Chat
git clone https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B

```

After downloading the models, where_you_download_dir should have the following structure:
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-Chat
├── Step-Audio-TTS-3B
```

###  🚀 4.3 Inference Scripts
#### Offline inference
Inference with e2e audio/text input and audio/text output.
```bash
python offline_inference.py --model-path where_you_download_dir
```
#### Launch Web Demo
Start a local server for online inference.
Assume you have 4 GPUs available and have already downloaded all the models.

```bash
python app.py --model-path where_you_download_dir
```

## 5. Benchmark

### 5.1 ASR result comparison

<table>
    <thead>
        <tr>
            <th style="text-align:center"></th>
            <th colspan="4" style="text-align:center">Hidden Feature Modeling</th>
            <th colspan="5" style="text-align:center">Discrete Audio Token Modeling</th>
        </tr>
        <tr>
            <th style="text-align:center"></th>
            <th style="text-align:center">Whisper Large-v3</th>
            <th style="text-align:center">Qwen2-Audio</th>
            <th style="text-align:center">MinMo</th>
            <th style="text-align:center">LUCY</th>
            <th style="text-align:center">Moshi</th>
            <th style="text-align:center">GLM-4-voice Base</th>
            <th style="text-align:center">GLM-4-voice Chat</th>
            <th style="text-align:center">Step-Audio Pretrain</th>
            <th style="text-align:center">Step-Audio-Chat</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Aishell-1</td>
            <td style="text-align:center">5.14</td>
            <td style="text-align:center">1.53</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">2.4</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">2.46</td>
            <td style="text-align:center">226.47</td>
            <td style="text-align:center"><strong>0.87</strong></td>
            <td style="text-align:center">1.95</td>
        </tr>
        <tr>
            <td>Aishell-2 ios</td>
            <td style="text-align:center">4.76</td>
            <td style="text-align:center">3.06</td>
            <td style="text-align:center"><strong>2.69</strong></td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">211.3</td>
            <td style="text-align:center">2.91</td>
            <td style="text-align:center">3.57</td>
        </tr>
        <tr>
            <td>Wenetspeech test-net</td>
            <td style="text-align:center">9.68</td>
            <td style="text-align:center">7.72</td>
            <td style="text-align:center"><strong>6.64</strong></td>
            <td style="text-align:center">8.78</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">146.05</td>
            <td style="text-align:center">7.62</td>
            <td style="text-align:center">8.75</td>
        </tr>
        <tr>
            <td>Wenet test-meeting</td>
            <td style="text-align:center">18.54</td>
            <td style="text-align:center">8.4</td>
            <td style="text-align:center"><strong>7.6</strong></td>
            <td style="text-align:center">10.42</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">140.82</td>
            <td style="text-align:center">7.78</td>
            <td style="text-align:center">9.52</td>
        </tr>
        <tr>
            <td>Librispeech test-clean</td>
            <td style="text-align:center">1.9</td>
            <td style="text-align:center"><strong>1.6</strong></td>
            <td style="text-align:center"><strong>1.6</strong></td>
            <td style="text-align:center">3.36</td>
            <td style="text-align:center">5.7</td>
            <td style="text-align:center">2.82</td>
            <td style="text-align:center">75.39</td>
            <td style="text-align:center">2.36</td>
            <td style="text-align:center">3.11</td>
        </tr>
        <tr>
            <td>Librispeech test-other</td>
            <td style="text-align:center">3.65</td>
            <td style="text-align:center"><strong>3.6</strong></td>
            <td style="text-align:center">3.82</td>
            <td style="text-align:center">8.05</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">7.66</td>
            <td style="text-align:center">80.3</td>
            <td style="text-align:center">6.32</td>
            <td style="text-align:center">8.44</td>
        </tr>
        <tr>
            <td>AVG</td>
            <td style="text-align:center">7.28</td>
            <td style="text-align:center"><strong>4.32</strong></td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">146.74</td>
            <td style="text-align:center">4.64</td>
            <td style="text-align:center">5.89</td>
        </tr>
    </tbody>
</table>

### 5.2 TTS
#### 5.2.1 Performance comparison of content consistency (CER/WER) between GLM-4-Voice and MinMo.

<table>
    <thead>
        <tr>
            <th rowspan="2">Model</th>
            <th style="text-align:center" colspan="1">test-zh</th>
            <th style="text-align:center" colspan="1">test-en</th>
        </tr>
        <tr>
            <th style="text-align:center">CER (%) &darr;</th>
            <th style="text-align:center">WER (%) &darr;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GLM-4-Voice</td>
            <td style="text-align:center">2.19</td>
            <td style="text-align:center">2.91</td>
        </tr>
        <tr>
            <td>MinMo</td>
            <td style="text-align:center">2.48</td>
            <td style="text-align:center">2.90</td>
        </tr>
        <tr>
            <td><strong>Step-Audio</strong></td>
            <td style="text-align:center"><strong>1.53</strong></td>
            <td style="text-align:center"><strong>2.71</strong></td>
        </tr>
    </tbody>
</table>

#### 5.2.2 Results of TTS Models on SEED Test Sets.
* StepAudio-TTS-3B-Single denotes dual-codebook backbone with single-codebook vocoder*

<table>
    <thead>
        <tr>
            <th rowspan="2">Model</th>
            <th style="text-align:center" colspan="2">test-zh</th>
            <th style="text-align:center" colspan="2">test-en</th>
        </tr>
        <tr>
            <th style="text-align:center">CER (%) &darr;</th>
            <th style="text-align:center">SS &uarr;</th>
            <th style="text-align:center">WER (%) &darr;</th>
            <th style="text-align:center">SS &uarr;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FireRedTTS</td>
            <td style="text-align:center">1.51</td>
            <td style="text-align:center">0.630</td>
            <td style="text-align:center">3.82</td>
            <td style="text-align:center">0.460</td>
        </tr>
        <tr>
            <td>MaskGCT</td>
            <td style="text-align:center">2.27</td>
            <td style="text-align:center">0.774</td>
            <td style="text-align:center">2.62</td>
            <td style="text-align:center">0.774</td>
        </tr>
        <tr>
            <td>CosyVoice</td>
            <td style="text-align:center">3.63</td>
            <td style="text-align:center">0.775</td>
            <td style="text-align:center">4.29</td>
            <td style="text-align:center">0.699</td>
        </tr>
        <tr>
            <td>CosyVoice 2</td>
            <td style="text-align:center">1.45</td>
            <td style="text-align:center">0.806</td>
            <td style="text-align:center">2.57</td>
            <td style="text-align:center">0.736</td>
        </tr>
        <tr>
            <td>CosyVoice 2-S</td>
            <td style="text-align:center">1.45</td>
            <td style="text-align:center">0.812</td>
            <td style="text-align:center">2.38</td>
            <td style="text-align:center">0.743</td>
        </tr>
        <tr>
            <td><strong>Step-Audio-TTS-3B-Single</strong></td>
            <td style="text-align:center">1.37</td>
            <td style="text-align:center">0.802</td>
            <td style="text-align:center">2.52</td>
            <td style="text-align:center">0.704</td>
        </tr>
        <tr>
            <td><strong>Step-Audio-TTS-3B</strong></td>
            <td style="text-align:center"><strong>1.31</strong></td>
            <td style="text-align:center">0.733</td>
            <td style="text-align:center"><strong>2.31</strong></td>
            <td style="text-align:center">0.660</td>
        </tr>
        <tr>
            <td><strong>Step-Audio-TTS</strong></td>
            <td style="text-align:center"><strong>1.17</strong></td>
            <td style="text-align:center">0.73</td>
            <td style="text-align:center"><strong>2.0</strong></td>
            <td style="text-align:center">0.660</td>
        </tr>
    </tbody>
</table>

#### 5.2.3 Performance comparison of Dual-codebook Resynthesis with Cosyvoice.

<table>
    <thead>
        <tr>
            <th style="text-align:center" rowspan="2">Token</th>
            <th style="text-align:center" colspan="2">test-zh</th>
            <th style="text-align:center" colspan="2">test-en</th>
        </tr>
        <tr>
            <th style="text-align:center">CER (%) &darr;</th>
            <th style="text-align:center">SS &uarr;</th>
            <th style="text-align:center">WER (%) &darr;</th>
            <th style="text-align:center">SS &uarr;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:center">Groundtruth</td>
            <td style="text-align:center">0.972</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">2.156</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td style="text-align:center">CosyVoice</td>
            <td style="text-align:center">2.857</td>
            <td style="text-align:center"><strong>0.849</strong></td>
            <td style="text-align:center">4.519</td>
            <td style="text-align:center"><strong>0.807</strong></td>
        </tr>
        <tr>
            <td style="text-align:center">Step-Audio-TTS-3B</td>
            <td style="text-align:center"><strong>2.192</strong></td>
            <td style="text-align:center">0.784</td>
            <td style="text-align:center"><strong>3.585</strong></td>
            <td style="text-align:center">0.742</td>
        </tr>
    </tbody>
</table>

### 5.3 AQTA Chat
We release [**StepEval-Audio-360**](https://huggingface.co/datasets/stepfun-ai/StepEval-Audio-360) as a new benchmark, which consists of 100 multi-turn Chinese prompts sourced from real users and is designed to evaluate the quality of generated response across the following dimensions: linguistic competence, emotional intelligence, logical reasoning, creative generation, commonsense knowledge, complex instruction adherence, role-playing capabilities, gaming interactions, real-world task handling, memory retention, safety compliance, educational tutoring, and comprehensive understanding.

#### 5.3.1 StepEval-Audio-360

#### LLM judge metrics(GPT-4o)
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th style="text-align:center">factuality (%) &uarr;</th>
            <th style="text-align:center">relevance (%) &uarr;</th>
            <th style="text-align:center">chat score &uarr;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GLM4-Voice</td>
            <td style="text-align:center">44.0</td>
            <td style="text-align:center">59.0</td>
            <td style="text-align:center">3.27</td>
        </tr>
        <tr>
            <td>Qwen2-Audio</td>
            <td style="text-align:center">25.0</td>
            <td style="text-align:center">28.0</td>
            <td style="text-align:center">2.23</td>
        </tr>
        <tr>
            <td>Moshi*</td>
            <td style="text-align:center">1.0</td>
            <td style="text-align:center">0</td>
            <td style="text-align:center">1.49</td>
        </tr>
        <tr>
            <td><strong>Step-Audio-Chat</strong></td>
            <td style="text-align:center"><strong>63.0</strong></td>
            <td style="text-align:center"><strong>73.0</strong></td>
            <td style="text-align:center"><strong>3.91</strong></td>
        </tr>
    </tbody>
</table>

*Note: Moshi are marked with "\*" and should be considered for reference only. 

#### Radar Chart(Human Evaluation)
<img src="./assets/stepeval_radar_chart.png" width="600" alt="QR code">

#### 5.3.2 Public Test Set

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th style="text-align:center">Llama Question</th>
            <th style="text-align:center">Web Questions</th>
            <th style="text-align:center">TriviaQA*</th>
            <th style="text-align:center">ComplexBench</th>
            <th style="text-align:center">HSK-6</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GLM4-Voice</td>
            <td style="text-align:center">64.7</td>
            <td style="text-align:center">32.2</td>
            <td style="text-align:center">39.1</td>
            <td style="text-align:center">66.0</td>
            <td style="text-align:center">74.0</td>
        </tr>
        <tr>
            <td>Moshi</td>
            <td style="text-align:center">62.3</td>
            <td style="text-align:center">26.6</td>
            <td style="text-align:center">22.8</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td>Freeze-Omni</td>
            <td style="text-align:center">72.0</td>
            <td style="text-align:center">44.7</td>
            <td style="text-align:center">53.9</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td>LUCY</td>
            <td style="text-align:center">59.7</td>
            <td style="text-align:center">29.3</td>
            <td style="text-align:center">27.0</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td>MinMo</td>
            <td style="text-align:center">78.9</td>
            <td style="text-align:center">55.0</td>
            <td style="text-align:center">48.3</td>
            <td style="text-align:center">-</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td>Qwen2-Audio</td>
            <td style="text-align:center">52.0</td>
            <td style="text-align:center">27.0</td>
            <td style="text-align:center">37.3</td>
            <td style="text-align:center">54.0</td>
            <td style="text-align:center">-</td>
        </tr>
        <tr>
            <td><strong>Step-Audio-Chat</strong></td>
            <td style="text-align:center"><strong><i>81.0</i></strong></td>
            <td style="text-align:center"><strong>75.1</strong></td>
            <td style="text-align:center"><strong>58.0</strong></td>
            <td style="text-align:center"><strong>74.0</strong></td>
            <td style="text-align:center"><strong>86.0</strong></td>
        </tr>
    </tbody>
</table>

*Note: Results marked with "\*" on TriviaQA dataset are considered for reference only.*

*TriviaQA dataset marked with "\*" indicates results are for reference only.*

## 6. Online Engine
The online version of Step-Audio can be accessed from app version of [跃问](https://yuewen.cn), where some impressive examples can be found as well.

<img src="./assets/yuewen.jpeg" width="200" alt="QR code">

## 7. Examples
### clone audio
| role   | prompt wav | clone wav |
|:-------:|:-------:|:-------:|
|赵本山| [audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/prompt_wav_zhaobenshan.mp3)|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/clone_wav_zhaobenshan.mp3)| 
|李雪琴| [audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/prompt_wav_lixueqin.wav)|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/clone_wav_lixueqin.mp3)| 

### speed control
| prompt | response |
|:-------:|:-------:|
|human: 说一个绕口令<br>assistant: 吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮<br>human: 哎，你能把这个绕口令说的再快一点吗？|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/speed_control1.wav)|
|human: 说一个绕口令<br>assistant: 吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮<br>human: 哎，你能把这个绕口令说的再快一点吗？<br>assistant: 吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮<br>human: 呃，你再用非常非常慢的速度说一遍的。|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/speed_control2.wav)|

### High EQ(emotional control & tone control)
| prompt | response |
|:-------:|:-------:|
|human: 你这语气又不撒娇又不卖萌的，要不你撒个娇卖个萌吧。|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/tone_control.wav)|
|human: 怎么办？我感觉我的人生很失败。|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/emotional_control1.wav)|
|human: 小跃。你真的是。特别厉害。|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/emotional_control2.wav)|

### Multilingual (e.g., Chinese, English, Japanese)
| prompt | response |
|:-------:|:-------:|
|English||
|human: human：こんにちは。（你好）<br>assistant：こんにちは！何か手伝いましょうか？（您好！我可以帮你做点什么吗？）|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/multilingual1.wav)|

### Rap & Vocal
| prompt | response |
|:-------:|:-------:|
|human：唱一段rap|[audio](https://github.com/stepfun-ai/Step-Audio/tree/gz/dev/examples/rap.wav)|

## 8. Citation
```
@misc{stepaudiotechnicalreport,
      title={Step-Audio: Harmonized Understanding and Generation for Intelligent Speech Interaction}, 
      author={Step-Audio Team},
      year={2025},
}
```
