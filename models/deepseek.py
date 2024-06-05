import torch
from models.deepseek_vl.models.modeling_vlm import MultiModalityCausalLM

from models.deepseek_vl.models import VLChatProcessor
from models.deepseek_vl.utils.io import load_pil_images

from models.base import Mllm

class DeepSeek(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_name_or_path
        )
        self.tokenizer = self.processor.tokenizer
        from transformers import AutoModelForCausalLM
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code = True
        ).to(torch.bfloat16).cuda().eval()
        
    def evaluate(self, prompt, filepath):
        conversation = [
            {
                'role': 'User',
                'content': f'<image_placeholder>{prompt}',
                'images': [filepath]
            },
            {
                'role': 'Assistant',
                'content': ''
            }
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations = conversation,
            images = pil_images,
            force_batchify = True
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(
            **prepare_inputs
        )
        outputs = self.model.language_model.generate(
            inputs_embeds = inputs_embeds,
            attention_mask = prepare_inputs.attention_mask,
            pad_token_id = self.tokenizer.eos_token_id,
            bos_token_id = self.tokenizer.bos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            max_new_tokens = 128,
            do_sample = False,
            use_cache = True 
        )
        response = self.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens = True
        )
        return response