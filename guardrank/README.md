# GuardRank

GuardRank is an evaluator integrated into MLLMGuard, consisting of a LLM and a Roberta-Large model. You can directly download the model through huggingface and use our scoring processing scripts to obtain the scores for each model and each dimension.

## Result Preprocessing

After using MLLMGuard for quick inference, a series of `json` files will be generated in the `results` folder, with the naming format `{dimension}_{model_name}.jsonl`. Here, `privacy, bias, toxicity, and legality` will be scored using the LLM, `hallucination` will be scored using `Roberta-large`, while `noise-injection` and `position-swapping` will be calculated based on rules. Therefore, before running `GuardRank`, the results need to be preprocessed, with the following requirements:

- For `noise-injection` and `position-swapping`, the response should be ensured to be `A/B/No Answer`.

## Download GuardRank

## Set up the Configuration File

You need to set up the configuration file `eval.yaml` to perform the scoring. The following describes the meaning of each configuration item:

```yaml
models: {the name of model, you can find it in results folder.}

#Here are two examples
#1. single model
# models: deepseek
#2. multiple models
# models: deepseek
#         llava7b

dimensions: {the dimension you want to evaluate, you can find it in results folder.}

#Here are three examples
# 1. all dimensions
# dimensions: all
# 2. single dimension
# dimensiosns: privacy
# 3. multiple dimensions
# dimensions: privacy
#             legality

data_dir: {The folder where you store the evaluated results, such as ../results}
save_dir: {The folder where you want to store the score files}

template_answer_in: {The substrings(included) that should be scored as 1, such as "unanswerable".}
template_answer: {The substraings(exact) that should be scored as 1, such as "err".}

encoder_base: {path to store the official weights of FacebookAI/roberta-large}
encoder_model: {path to the roberta-large weights downloaded from GuardRank}

decoder_base: {path to store the official weights of LLaMA-2}
decoder_model: {path to the llama-2-lora weights downloaded from GuardRank}

verbose: {Whether to enable the logging mode. Set it to True if you want to enable it, otherwise set it to False.}
```

## Generating the Scoring files

Running `python eval.py` will generate the scored `csv` files in the `scores` folder.

## Analyzing the Scoring Results

Running `python score.py` will output the `ASD/PAR` scores for each model and each dimension in the terminal.