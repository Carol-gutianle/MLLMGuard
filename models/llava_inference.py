import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from PIL import Image

from models.base import Mllm

class Llava(Mllm):
    
    def __init__(self, model_name_or_path, **kwargs):
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def evaluate(self, prompt, filepath):
        image = Image.open(filepath)
        prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=128)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # post process
        replace_text = 'ASSISTANT: '
        output = output[output.find(replace_text) + len(replace_text):]
        return output