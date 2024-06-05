from PIL import Image

import torch

from models.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from models.mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from models.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from models.base import Mllm


class mPLUG_Owl(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype = torch.bfloat16
        )
        processor = MplugOwlImageProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(model_name_or_path)
        self.processor = MplugOwlProcessor(processor, self.tokenizer)
    
    def evaluate(self, prompt, filepath):
        image_list = [filepath]
        prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: {prompt}
        AI: ''']
        generate_kwargs = {
            'do_sample': False,
            'max_length': 128
        }
        images = [Image.open(_) for _ in image_list]
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence = self.tokenizer.decode(
            res.tolist()[0],
            skip_special_tokens = True
        )
        return sentence

from models.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from models.mplug_owl2.conversation import conv_templates
from models.mplug_owl2.model.builder import load_pretrained_model
from models.mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class mPLUG_Owl2(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        model_name = get_model_name_from_path(model_name_or_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_name_or_path,
            None,
            model_name,
            load_8bit = False,
            load_4bit = False,
            device_map = None,
            device = 'cuda'
        )
        self.model = self.model.to('cuda')
        self.conv = conv_templates['mplug_owl2'].copy()
        self.roles = self.conv.roles
    
    def evaluate(self, prompt, filepath):
        image = Image.open(filepath).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))
        
        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, torch.float16)
        
        self.conv = conv_templates['mplug_owl2'].copy()
        inp = DEFAULT_IMAGE_TOKEN + prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        
        prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)
        stop_str = self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        temperature = 0
        max_new_tokens = 128
        
        with torch.inference_mode():
            try:
                output_ids = self.model.generate(
                    input_ids,
                    images = image_tensor,
                    do_sample = False,
                    temperature = temperature,
                    top_k = 1,
                    max_new_tokens = max_new_tokens,
                    use_cache = False,
                    stopping_criteria = [stopping_criteria]
                )
            except Exception as e:
                traceback.print_exc()
                    
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs = outputs.replace('<|endoftext|>', '')
        return outputs