import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.base import Mllm

class Xcomposer(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True).cuda().eval()
    
    def evaluate(self, prompt, filepath):
        torch.set_grad_enabled(False)
        prompt = '<ImageHere>' + prompt
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(
                self.tokenizer,
                query = prompt,
                image = filepath,
                history = [],
                do_sample = False
            )
        return response