import sys
sys.path.append('..')

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from PIL import Image

from models.base import Mllm

class CogVLM(Mllm):
    def __init__(self, model_name_or_path, tokenizer_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype = torch.float16,
                low_cpu_mem_usage = True,
                trust_remote_code = True
            )
        device_map = infer_auto_device_map(self.model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
        self.model = load_checkpoint_and_dispatch(
            self.model,
            model_name_or_path,
            device_map=device_map,
        ).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        
    def evaluate(self, prompt, filepath):
        image = Image.open(filepath).convert('RGB')
        query = prompt
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], images=[image], template_version='vqa')
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.float16)]],
        }
        
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            
        return response