# GuardRank

GuardRank是集成在MLLMGuard中的评估器，由一个LLM和一个Roberta-large组成。您可以直接通过链接下载打分模型，并且使用我们的得分处理文件获得各模型，各维度的得分。

## 结果预处理

当您使用MLLMGuard进行快速推理后，会在`results`文件夹生成一系列`jsonl`文件，其命名格式为`{dimension}_{model_name}.jsonl`。其中`privacy, bias, toxicity以及legality`会使用LLM进行评分，`hallucination`会使用`Roberta-large`进行评分，而`noise-injection`和`position-swapping`将根据规则计算得分。所以请在GuardRank打分前，进行结果的预处理，要求如下：

- 对于noise-injection和position-swapping，需要确保response为A/B/No Answer.

## 下载GuardRank

## 设置配置文件

您需要通过设置配置文件`eval.yaml`进行打分，下面介绍各个配置项的含义。

```yaml
models: {模型名称，可以参考results文件夹。}

#这里有两个例子
#1. 单个模型
# models: deepseek
#2. 多个模型
# models: deepseek
#         llava7b

dimensions: {评测维度，可以参考results文件夹。}

#这里有三个例子
# 1. 所有维度
# dimensions: all
# 2. 单个维度
# dimensiosns: privacy
# 3. 多个维度
# dimensions: privacy
#             legality

data_dir: {存储推理结果的文件夹，比如../results}
save_dir: {想要保存的得分的文件夹，比如scores}

template_answer_in: {需要设置为得分为1的包含子串，比如unanswerable}
template_answer: {需要设置为得分为1的相等子串，比如err}

encoder_base: {存储`FacebookAI/roberta-large`官方权重的位置}
encoder_model: {下载的GuarRank中Roberta-large的权重位置}

decoder_base: {存储`LLaMA-2`官方权重的位置}
decoder_model: {下载的GuardRank中LLaMA-2-LoRA的权重位置}

verbose: {是否打开日志模式，如果是则设置为True，否则设置为False}
```

## 生成打分文件

运行`python eval.py`，将在`scores`文件夹下生成打分后的`csv`文件。

## 分析打分结果

运行`python score.py`，将在终端输出各模型/各维度的ASD/PAR得分。
