# MLLMGuard

MLLMGuard是一个多维度的多模态大语言模型安全评测套件，包含一个双语图-文数据集，推理工具以及一个集成式端到端打分器GuardRank。

[![Github Repo Stars](https://img.shields.io/github/stars/Carol-gutianle/MLLMGuard?style=social)](https://github.com/Carol-gutianle/MLLMGuard/stargazers)
[![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Data-yellow)](https://huggingface.co/datasets/Carol0110/MLLMGuard)
[![GitHub issues](https://img.shields.io/github/issues/Carol-gutianle/MLLMGuard)](https://github.com/Carol-gutianle/MLLMGuard/issues)
[![arXiv](https://img.shields.io/badge/arXiv-en-red)](https://arxiv.org/abs/2406.07594)

## 目录

- [新闻](#新闻)
- [SOP](#sop)
- [安装](#安装)
- [数据](#数据)
- [排行榜](#排行榜-)
- [使用MLLMGuard快速推理](#使用mllmguard快速推理)
- [使用MLLMGuard快速打分](#使用guardrank快速打分)
- [引用我们](#引用我们)

## 新闻

[2024-06-06]我们发布了MLLMGuard评估套件。

## SOP

1. [安装环境](#安装)
2. [下载数据](#数据)
3. [快速推理](#使用mllmguard快速推理)
4. [快速打分](#使用guardrank快速打分)

## 安装

```bash
git clone https://github.com/Carol-gutianle/MLLMGuard.git
conda create -n guard python=3.10
pip install -r requirements.txt
cd MLLMGuard
# 创建新的文件夹
mkdir {data, results, logs}
```

请将下载后的数据存放在data文件夹下，data文件夹的格式为：

```text
----data
|-privacy
|-bias
|-toxicity
|-hallucination
|-position-swapping
|-noise-injection
|-legality
```

## 数据

### 下载

您可以在Hugging Face[Huggin Face Website](https://huggingface.co/datasets/Carol0110/MLLMGuard)中获取MLLMGuard数据集。

### 描述

目前开源的数据集为MLLMGuard(Public)版本，共1,500条数据，并且经过脱敏处理。如果您需要未脱敏数据进行评测，请在这里填写表单。

## 排行榜 🏆

您可以在HuggingFace Space[Hugging Face Space](https://huggingface.co/spaces/Carol0110/MLLMGuardLeaderboard)查看最新榜单。

## 使用MLLMGuard快速推理

你可以在scripts中找到模型的推理脚本，我们提供了闭源模型和开源模型的推理脚本。

### 闭源模型

我们提供了闭源模型的评估脚本`evaluate_api.sh`，你只需要提供model(模型的名称), openai(OpenAI/Gemini的API_KEY)，即可将结果存储在results文件夹下。

以下是已经实现的模型以及对应的在model中的简称：

- GPT-4V: gpt4v
- Yi-VL-Plus: yi-vl-plus
- Gemini: gemini

### 开源模型

我们也提供了开源模型的评估脚本`eval.sh`，你只需要提供model(模型的名称/路径), category(评估的维度)，即可将结果存储在results文件夹下。

以下是已经实现的模型：

- LLaVA-v1.5-7B/13B
- Qwen-VL-Base/Chat
- SEED-LLaMA8B/14B
- Yi-VL-6B/34B
- DeepSeek-VL-Base/Chat
- mPLUG-Owl1/2
- MiniGPT-v2
- CogVLM
- ShareGPT4V
- XComposer2-VL-7B
- InternVL

### 自定义模型

你可以通过继承`models/base.py`中的`Mllm`类并重写`__init__`和`evaluate`函数来添加自己的模型。请记得在`evaluate.py`中添加自定义模型的接口，具体样例可以参考`cogvlm.py`以及下面的代码片段：

```python
        elif 'cogvlm' in model_name:
            from models.cogvlm import CogVLM
            cogvlm = CogVLM(args.model, args.tokenizer)
            evaluate_model(cogvlm, args, data)
```

## 使用GuardRank快速打分

你可以使用GuardRank进行快速打分，过程请[参考](./guardrank/README_ZH.md)。

## 引用我们

如果您对MLLMGuard感兴趣，可以以下面的方式引用我们。

```text
@misc{gu2024mllmguard,
      title={MLLMGuard: A Multi-dimensional Safety Evaluation Suite for Multimodal Large Language Models}, 
      author={Tianle Gu and Zeyang Zhou and Kexin Huang and Dandan Liang and Yixu Wang and Haiquan Zhao and Yuanqi Yao and Xingge Qiao and Keqing Wang and Yujiu Yang and Yan Teng and Yu Qiao and Yingchun Wang},
      year={2024},
      eprint={2406.07594},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
