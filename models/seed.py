import torch

from PIL import Image

from models.seed_llama.model_tools import get_pretrained_llama_causal_model
from models.seed_llama.transforms import get_transform
from models.seed_llama.seed_llama_tokenizer import SeedLlamaTokenizer
from models.base import Mllm

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<img>'
NUM_IMG_TOKENS = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000
        
class SeedLLaMA8B(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model = get_pretrained_llama_causal_model(
            pretrained_model_name_or_path = model_name_or_path,
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True   
        ).eval().to(self.device)
        self.tokenizer = SeedLlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path = "/mnt/cachenew/gutianle/AILab-CVC/seed-tokenizer-2",
            fp16 = True,
            load_diffusion = False,
            encoder_url = "/mnt/cachenew/gutianle/AILab-CVC/seed-tokenizer-2/seed_quantizer.pt",
            diffusion_path = "/mnt/cachenew/gutianle/stabilityai/stable-diffusion-2-1-unclip",
            device = self.device,
            legacy = False
        )
        self.transform = get_transform(
            type = "clip",
            image_size = 224,
            keep_ratio = False
        )
    
    def evaluate(self, prompt, filepath):
        
        def __generate__(tokenizer, input_tokens, generation_config, model):
            input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
            input_ids = input_ids.to('cuda')
            
            generate_ids = model.generate(
                input_ids = input_ids,
                **generation_config
            )
            
            generate_ids = generate_ids[0][input_ids.shape[1]:]
            return generate_ids
        
        def __decode__(generate_ids):
            boi_list = torch.where(
                generate_ids == self.tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0]
            )[0]
            eoi_list = torch.where(
                generate_ids == self.tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0]
            )[0]
            if len(boi_list) == 0 and len(eoi_list) == 0:
                text_ids = generate_ids
                texts = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            else:
                boi_index = boi_list[0]
                text_ids = generate_ids[:boi_index]
                if len(text_ids) != 0:
                    texts = self.tokenizer.decode(
                        text_ids,
                        skip_special_tokens = True
                    )
            return texts
        
        generation_config = {
            'temperature': 0.0,
            'num_beams': 1,
            'max_new_tokens': 128,
            'top_p': 0,
            'do_sample': False
        }
        
        s_token = 'USER:'
        e_token = 'ASSISTANT:'
        sep = '\n'
        
        image = Image.open(filepath).convert('RGB')
        image_tensor = self.transform(image).to(self.device)
        image_ids = self.tokenizer.encode_image(image_torch = image_tensor)
        image_ids = image_ids.view(-1).cpu().numpy()
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in image_ids]) + EOI_TOKEN
        
        input_tokens = self.tokenizer.bos_token + s_token + " " + image_tokens + prompt + sep + e_token
        generate_ids = __generate__(self.tokenizer, input_tokens, generation_config, self.model)
        
        response = __decode__(generate_ids)
        
        return response
    
class SeedLLaMA14B(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_pretrained_llama_causal_model(
            low_cpu_mem_usage = True,
            torch_dtype = torch.float16,
            pretrained_model_name_or_path = model_name_or_path
        ).eval().to(self.device)
        self.tokenizer = SeedLlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path = "/mnt/cachenew/gutianle/AILab-CVC/seed-tokenizer-2",
            fp16 = True,
            load_diffusion = False,
            encoder_url = "/mnt/cachenew/gutianle/AILab-CVC/seed-tokenizer-2/seed_quantizer.pt",
            diffusion_path = "/mnt/cachenew/gutianle/stabilityai/stable-diffusion-2-1-unclip",
            device = self.device,
            legacy = False
        )
        self.transform = get_transform(
            type = "clip",
            image_size = 224,
            keep_ratio = False
        )
    
    def evaluate(self, prompt, filepath):
        
        s_token = '[INST]'
        e_token = '[/INST]'
        sep = '\n'
        
        def __generate__(input_tokens, generation_config):
            input_ids = self.tokenizer(
                input_tokens,
                add_special_tokens = False,
                return_tensors = 'pt'
            ).input_ids
            input_ids = input_ids.to(self.device)
            generate_ids = self.model.generate(
                input_ids = input_ids,
                **generation_config
            )
            generate_ids = generate_ids[0][input_ids.shape[1]:]
            return generate_ids
        
        def __decode__(generate_ids):
            response = ''
            
            boi_list = torch.where(
                generate_ids == self.tokenizer(
                    BOI_TOKEN,
                    add_special_tokens = False
                ).input_ids[0]
            )[0]
            eoi_list = torch.where(
                generate_ids == self.tokenizer(
                    EOI_TOKEN,
                    add_special_tokens = False
                ).input_ids[0]
            )[0]
            
            if len(boi_list) == 0 and len(eoi_list) == 0:
                text_ids = generate_ids
                response = self.tokenizer.decode(
                    text_ids,
                    skip_special_tokens = True
                )
            else:
                boi_idx = boi_list[0]
                text_ids = generate_ids[:boi_idx]
                if len(text_ids) != 0:
                    response = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            
            return response
        
        generation_config = {
            'temperature' : 0.0,
            'num_beams' : 1,
            'max_new_tokens' : 128,
            'top_p' : 0,
            'do_sample' : False
        }
        
        image = Image.open(filepath).convert('RGB')
        image_tensor = self.transform(image).to(self.device)
        image_ids = self.tokenizer.encode_image(image_torch = image_tensor)
        image_ids = image_ids.view(-1).cpu().numpy()
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in image_ids]) + EOI_TOKEN
        
        input_tokens = self.tokenizer.bos_token + s_token + image_tokens + prompt + e_token + sep
        
        generate_ids = __generate__(
            input_tokens,
            generation_config
        )
        
        response = __decode__(generate_ids)
        
        return response