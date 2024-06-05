from PIL import Image
import torch

from transformers import StoppingCriteriaList

from minigpt4.minigpt4.common.config import Config
from minigpt4.minigpt4.common.registry import registry
from minigpt4.minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.minigpt4.datasets.builders import *
from minigpt4.minigpt4.models import *
from minigpt4.minigpt4.processors import *
from minigpt4.minigpt4.runners import *
from minigpt4.minigpt4.tasks import *

from models.base import Mllm

class MiniGPTV2(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        cfg = Config(model_name_or_path)
        model_config = cfg.model_cfg
        model_config.device_8bit = 0
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(0))
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(0)) for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(0), stopping_criteria=self.stopping_criteria)
        self.model = model
        self.vis_processor = vis_processor

        
    def evaluate(self, prompt, filepath):
        chat_state = CONV_VISION_minigptv2.copy()
        # if use minigpt4, please change last line to: chat_state = CONV_VISON_vicuna0.copy()
        img_list = []
        llm_message = self.chat.upload_img(filepath, chat_state, img_list)
        
        self.chat.ask(prompt, chat_state)
        self.chat.encode_img(img_list)
        llm_message = self.chat.answer(
            conv = chat_state,
            img_list = img_list,
            max_new_tokens = 128,
            num_beams = 1,
            temperature = 0.1
        )[0]
        
        return llm_message