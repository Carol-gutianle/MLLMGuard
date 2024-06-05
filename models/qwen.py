import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from models.base import Mllm

class Qwen(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True).eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.to(self.device)
        
    def evaluate(self, prompt, filepath):
        prompt = self.tokenizer.from_list_format(
            [
                {'image': filepath},
                {'text': prompt}
            ]
        )
        response, _ = self.model.chat(self.tokenizer, query=prompt, history=None)
        return response

class QwenVL(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map = 'cuda',
            trust_remote_code = True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code = True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def evaluate(self, prompt, filepath):
        query = self.tokenizer.from_list_format([
            {'image': filepath},
            {'text': prompt}
        ])
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.device)
        preds = self.model.generate(**inputs)
        response = self.tokenizer.decode(preds.cpu()[0], skip_special_tokens=False)
        return response