# MLLMGuard

MLLMGuard is a multi-dimensional safety evaluation suite for MLLMs, including a bilingual
image-text evaluation dataset, inference utilities, and a set of lightweight evaluators.

[![Github Repo Stars](https://img.shields.io/github/stars/Carol-gutianle/MLLMGuard?style=social)](https://github.com/Carol-gutianle/MLLMGuard/stargazers)
[![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Data-yellow)](https://huggingface.co/datasets/Carol0110/MLLMGuard)
[![GitHub issues](https://img.shields.io/github/issues/Carol-gutianle/MLLMGuard)](https://github.com/Carol-gutianle/MLLMGuard/issues)
[![arXiv](https://img.shields.io/badge/arXiv-en-red)](https://arxiv.org/abs/2406.07594)

## Table of Contents

- [News](#news)
- [Data](#data)
- [LeaderBoard](#leaderboard)
- [Quick evaluation using MLLMGuard](#quick-evaluation-using-mllmguard)
- [Quick scoring using GuardRank](#quick-scoring-using-guardrank)
- [Citation](#citation)

## News

[2024-06-06]We release the safety evaluation suite MLLMGuard.

## SOP

1. [Installation](#installation)
2. [Data Download](#data)
3. [Quick Inference](#quick-evaluation-using-mllmguard)
4. [Quick Scoring](#quick-scoring-using-guardrank)

## Installation

```bash
git clone https://github.com/Carol-gutianle/MLLMGuard.git
conda create -n guard python=3.10
pip install -r requirements.txt
cd MLLMGuard
# create new folders
mkdir {data, results, logs}
```

Please put the downloaded data under the `data` folder. The format of the `data` folder is:

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

## Data

### Download

We put our data on [Huggin Face Website](https://huggingface.co/datasets/Carol0110/MLLMGuard)

### Description

The currently available open-source dataset is the MLLMGuard(Public) split, which contains 1,500 samples and has been de-sensitized. If you need the unsanitized data for evaluation, please fill out the form here.

## LeaderBoard

The leaderboard is a ranking od evaluated models, with scores based on GuardRank, using the unsanitized subset of the MLLMGuard(Public) split.

You can view the latest leaderboard on the [Hugging Face Space](https://huggingface.co/spaces/Carol0110/MLLMGuardLeaderboard).

## Quick Evaluation using MLLMGuard

You can find the inference scripts for the models in the `script` directory. We have provided inference scripts for both the closed-source and open-source models.

### Closed-source models

We have provided an evaluation script `evaluate_api.sh` for the closed-source mdoels. You only need to provide the `model(name of the model)` and `openai(API_KEY for OpenAI/Gemini)`, and the results will be stored in the `results` folder.

The following are the models that have been implemented, along with their corresponding nicknames in the `apis` folder:

- GPT-4V: gpt-4v
- Gemini: gemini

### Open-source models

We also provide an evaluation script `eval.sh` for the open-source models. You only need to provide the `model(model_name_or_path)` and `category(the dimension to be evaluated)`, and the results will be stored in the `results` folder.

### Your own model

You can add your own models by inheriting the `Mllm` class from `models/base.py` and overriding the `__init__` and `evaluate` functions. Remeber to add the interface for your own custom model in evaluate.py as well. You can refer to the example in `cogvlm.py` and the code snippet below:

```python
        elif 'cogvlm' in model_name:
            from models.cogvlm import CogVLM
            cogvlm = CogVLM(args.model, args.tokenizer)
            evaluate_model(cogvlm, args, data)
```

## Quick Scoring using GuardRank

You can use GuardRank for quick scoring. Please refer to [GuardRank](./guardrank/README.md) for the process.

## Citation

If you think this evaluation suite is helpful, please cite the paper.

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
