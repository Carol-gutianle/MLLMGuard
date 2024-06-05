import torch
from PIL import Image
import traceback

from share4v.model.builder import load_pretrained_model
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init

from models.base import Mllm

class ShareGPT(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path = model_name_or_path,
            model_base = None,
            model_name = get_model_name_from_path(model_name_or_path)
        )
        
    
    def evaluate(self, prompt, filepath):
        try:
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                        DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            conv_mode = "share4v_v0"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            image = Image.open(filepath).convert('RGB')
            image_tensor = self.image_processor.preprocess(
                image,
                return_tensors = 'pt'
            )['pixel_values'].half().cuda()
            
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors = 'pt'
            ).unsqueeze(0).cuda()
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords,
                self.tokenizer,
                input_ids
            )
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images = image_tensor,
                    do_sample = False,
                    max_new_tokens = 128,
                    use_cache = True,
                    stopping_criteria = [stopping_criteria]
                )
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]
            ).sum().item()
            
            if n_diff_input_output > 0:
                print(
                    f'[Warining] {n_diff_input_output} output_ids are not the same as the input_ids'
                )
            
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens = True
            )[0]
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
                
            outputs = outputs.strip()
            return outputs
        except Exception as e:
            traceback.print_exc()