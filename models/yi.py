import os
import torch
from PIL import Image
import traceback

from models.llava.conversation import conv_templates
from models.llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token
)
from models.llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info

from models.base import Mllm

class YIVL(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        try:
            model_path = os.path.expanduser(model_name_or_path)
            key_info['model_path'] = model_path
            get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path
            )
            self.model = self.model.to(dtype = torch.bfloat16)
        except Exception as e:
            traceback.print_exc()
        
    def _disable_torch_init(self):
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
    def evaluate(self, prompt, filepath):
        self._disable_torch_init()
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_templates['mm_default'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0)
            .cuda()
        )
        
        image = Image.open(filepath)
        if getattr(self.model.config, 'image_aspect_ratio', None) == 'pad':
            image = expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        image_tensor = self.image_processor.preprocess(
            image,
            return_tensors = 'pt'
        )['pixel_values'][0]
        
        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample = False,
                temperature = 0.1,
                num_beams = 1,
                stopping_criteria = [stopping_criteria],
                max_new_tokens = 128,
                use_cache = True
            )
        
        input_token_len = input_ids.shape[1]
        
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]
        ).sum().item()
        if n_diff_input_output != 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input ids!')
        
        response = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens = True
        )[0]
        response = response.strip()
        
        if response.endswith(stop_str):
            response = response[: -len(stop_str)]
            
        response = response.strip()
        
        return response