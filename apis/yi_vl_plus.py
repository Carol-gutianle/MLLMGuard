import time
import base64
from openai import OpenAI
import sys
sys.path.append('..')
from models import Mllm

class Yi_VL(Mllm):
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.client = OpenAI(
            api_key = model_name_or_path,
            base_url = "https://api.lingyiwanwu.com/v1"
        )
        
    def __call__(self, message):
        status = 0
        while status != 1:
            try:
                completion = self.client.chat.completions.create(
                    model = 'yi-vl-plus',
                    messages = message,
                    temperature = 0.1,
                    max_tokens = 128
                )
                RESPONSE = completion
                status = 1
                time.sleep(10)
            except Exception as e:
                print(e)
                time.sleep(10)
                raise ValueError(1)
        return RESPONSE
                
    def evaluate(self, prompt, filepath):
        def encode_image():
            with open(filepath, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        image = encode_image()
        message = [
            {
                'role': 'user',
                'content': [
                    {'type':'text', 'text':f'{prompt}'},
                    {'type':'image_url', 'image_url':{"url":f"data:image/jpeg;base64,{image}"}}
                ]
            }
        ]
        try:
            output = self.__call__(message)
        except:
            output = 'err'
        return output