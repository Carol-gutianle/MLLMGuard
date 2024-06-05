import time
import json
import openai
import base64
from tqdm import tqdm
import jsonlines

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPT4V:
    
    def __init__(self, api_key, engine="gpt-4-vision-preview", temperature=0, sleep_time=10) -> None:
        openai.api_key = api_key 
        self.engine = engine
        self.client = openai.OpenAI(api_key=api_key)
        self.temperature = temperature
        self.sleep_time = sleep_time
    
    def call(self, message):
        status = 0
        while status != 1:
            try:
                completion = self.client.chat.completions.create(
                    model = self.engine,
                    messages=message,
                    temperature=self.temperature,
                    max_tokens = 128
                )
                RESPONSE = completion.choices[0].message.content
                status = 1
                time.sleep(self.sleep_time)
            except Exception as e:
                print(e)
                time.sleep(10)
                raise ValueError(1)
        return RESPONSE
    
    def draw(self, prompt, save_path):
        """
        保存画出的图片
        模型只能提供dalle-2和dalle-3,或者不提供
        """
        response = self.client.images.generate(
            prompt = prompt,
            model = 'dall-e-3',
            n = 1,
            quality = 'standard',
            size = '1024x1024',
            style = 'vivid',
            response_format = 'url'
        )
        print(response)
        result = json.loads(response.json())
        image_url = result['data'][0]['url']
        import requests
        def save_image_by_url(url, file_name: str = "output.png"):
            response = requests.get(url)
            with open(file_name, "wb") as f:
                f.write(response.content)
        result = json.loads(response.json())
        image_url = result["data"][0]["url"]
        save_image_by_url(image_url, save_path)
        
    def evaluate(self, prompt, image):
        base64_image = encode_image(image)
        message = [
            {
                'role': 'user',
                'content': [
                    {'type':'text', 'text':f'{prompt}'},
                    {'type':'image_url', 'image_url':{"url":f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        try:
            output = self.call(message)
        except:
            output = 'err'
        return output
        
    def batch_evaluate(self, data, args):
        response = []
        for sample in tqdm(data):
            base64_image = encode_image(sample['img_url'])
            prompt = sample['prompt']
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type':'text', 'text':f'{prompt}'},
                        {'type':'image_url', 'image_url':{"url":f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            try:
                res = self.call(message)
                result = {'prompt':prompt, 'img_url':sample['img_url'], 'response': res, 'lan': sample['lan']}
                if args.verbose:
                    print(result)
                response.append(result)
            except:
                response.append({'prompt':prompt, 'img_url':sample['img_url'], 'response': 'err', 'lan': sample['lan']})
                
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response)