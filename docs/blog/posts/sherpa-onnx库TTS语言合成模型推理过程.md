---
title: sherpa-onnx库TTS语言合成模型推理过程
date: 2025-02-17
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
comments: true
---

<!-- more -->

## 两种模型架构的区别

从模型推理的角度看，vits对应了一个onnx文件，文字直接输出音频；而matcha-tts模型有两个onnx文件，先将文字转换梅尔频谱，然后采用声码器（比如hifigan）生成音频。

```
引用自deepseek

Matcha-TTS与VITS作为两种主流的语音合成（TTS）模型，在架构设计和性能表现上存在显著差异，具体对比如下：

一、架构设计

Matcha-TTS‌

基于编码器-解码器结构，采用‌条件流匹配（OT-CFM）‌进行训练，通过最优传输理论优化数据转换过程。
解码器基于常微分方程（ODE）构建，支持更少的合成步骤生成高质量语音。
非自回归设计‌，无需依赖外部对齐机制，直接学习文本到语音的映射。

VITS‌

完全端到端模型，直接将文本映射为语音波形，‌无需中间声学特征提取‌（如梅尔频谱）。
结合‌变分推理与对抗训练‌，通过随机潜在变量建模语音的自然度和多样性。
支持语音克隆、流式推理等扩展能力，可通过微调实现个性化音色生成。
二、生成方式与效率
特性	Matcha-TTS	VITS
生成方式‌	基于ODE的概率采样	基于随机潜在变量生成
合成步骤‌	少步骤（10步内）	需要更多步骤优化生成质量
推理速度‌	CPU/GPU均高效，适合实时场景	依赖GPU加速，端到端效率高
内存占用‌	较低	较高（尤其长语音场景）
三、性能表现

Matcha-TTS‌

在合成速度上接近实时模型，长语音场景表现优异。
主观评测（MOS）得分较高，尤其在音质和自然度上优于传统扩散模型。

VITS‌

生成语音的自然度和情感表达更接近真实人声，尤其在微调后表现突出。
社区生态完善，支持多语言、歌声合成、语音克隆等多种扩展场景。
四、应用场景
Matcha-TTS‌：适合对‌实时性要求高‌的场景（如对话系统、直播字幕转语音）。
VITS‌：适用于需要‌高自然度与多场景扩展性‌的任务（如虚拟主播、个性化语音克隆）。
五、技术演进趋势
Matcha-TTS‌代表扩散模型的优化方向，通过流匹配减少计算量，未来可能进一步压缩推理步骤。
VITS‌持续扩展功能边界（如结合LLM优化文本理解能力），并在开源社区推动模型轻量化与多语言支持。

总结：Matcha-TTS在速度与资源占用上占优，而VITS在自然度与功能多样性上更胜一筹，两者适用场景互补。
```


## sherpa-onnx推理流程以及改造成硬件加速推理

https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts.py中的使用示例：

```bash
Example (3/7)

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
rm sherpa-onnx-vits-zh-ll.tar.bz2

python3 ./python-api-examples/offline-tts.py \
 --vits-model=./sherpa-onnx-vits-zh-ll/model.onnx \
 --vits-lexicon=./sherpa-onnx-vits-zh-ll/lexicon.txt \
 --vits-tokens=./sherpa-onnx-vits-zh-ll/tokens.txt \
 --tts-rule-fsts=./sherpa-onnx-vits-zh-ll/phone.fst,./sherpa-onnx-vits-zh-ll/date.fst,./sherpa-onnx-vits-zh-ll/number.fst \
 --vits-dict-dir=./sherpa-onnx-vits-zh-ll/dict \
 --sid=2 \
 --output-filename=./test-2.wav \
 "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。2024年5月11号，拨打110或者18920240511。123456块钱。"

```

```bash
Example (4/7)

curl -O -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

python3 ./python-api-examples/offline-tts.py \
 --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
 --matcha-vocoder=./vocos-22khz-univ.onnx \
 --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
 --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
 --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
 --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
 --output-filename=./test-matcha.wav \
 "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"
```

通过分析sherpa-onnx推理逻辑，改造成硬件加速推理。将onnxruntime推理替换成硬件runtime推理引擎（包括模型转换、推理加速）。
查看onnx文件的metadata, 
```
/project/sherpa-onnx/csrc/offline-tts-vits-model.cc:Init:151 ---vits model---
description=MeloTTS is a high-quality multi-lingual text-to-speech library by MyShell.ai
license=MIT license
url=https://github.com/myshell-ai/MeloTTS
model_type=melo-vits
version=2
sample_rate=44100
add_blank=1
n_speakers=1
jieba=1
bert_dim=1024
language=Chinese + English
ja_bert_dim=768
speaker_id=1
comment=melo
lang_id=3
tone_start=0
----------input names----------
0 x
1 x_lengths
2 tones
3 sid
4 noise_scale
5 length_scale
6 noise_scale_w
----------output names----------
0 y
```

首先，根据metadata可以得出，采用了frontend 将text转换成tokens。

```c++
https://github1s.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-vits-impl.h#L210-L211
    std::vector<TokenIDs> token_ids =
        frontend_->ConvertTextToTokenIds(text, meta_data.voice);
```
MeloTtsLexicon 作为frontend。
```c++
else if (meta_data.jieba && !config_.model.vits.dict_dir.empty() &&
               meta_data.is_melo_tts) {
    frontend_ = std::make_unique<MeloTtsLexicon>(
        mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
        config_.model.vits.dict_dir, model_->GetMetaData(),
        config_.model.debug);
} 
```
看一下推理整个调用栈：
```c++
// https://sourcegraph.com/github.com/k2-fsa/sherpa-onnx@f00066db88853852c68eed81a8686cd27d934240/-/blob/sherpa-onnx/csrc/offline-tts-impl.h?L26

virtual GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const = 0;


// https://sourcegraph.com/github.com/k2-fsa/sherpa-onnx@f00066db88853852c68eed81a8686cd27d934240/-/blob/sherpa-onnx/csrc/offline-tts-vits-impl.h?L152
  GeneratedAudio Generate(
      const std::string &_text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;
    ...
    }

// https://sourcegraph.com/github.com/k2-fsa/sherpa-onnx@f00066db88853852c68eed81a8686cd27d934240/-/blob/sherpa-onnx/csrc/offline-tts-vits-impl.h?L209

    std::vector<TokenIDs> token_ids =
        frontend_->ConvertTextToTokenIds(text, meta_data.voice);

// https://sourcegraph.com/github.com/k2-fsa/sherpa-onnx@f00066db88853852c68eed81a8686cd27d934240/-/blob/sherpa-onnx/csrc/offline-tts-vits-impl.h?L249
    if (config_.max_num_sentences <= 0 || x_size <= config_.max_num_sentences) {
      auto ans = Process(x, tones, sid, speed);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size(), 1.0);
      }
      return ans;
    }

// https://sourcegraph.com/github.com/k2-fsa/sherpa-onnx@f00066db88853852c68eed81a8686cd27d934240/-/blob/sherpa-onnx/csrc/offline-tts-vits-impl.h?L430
  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         const std::vector<std::vector<int64_t>> &tones,
                         int32_t sid, float speed) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

......
                         }


```