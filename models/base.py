import sys
sys.path.append('..')

from tqdm import tqdm
import jsonlines

from utils import RESPONSE_DICT

class Mllm:
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        pass
    
    def evaluate(self, prompt, filepath):
        pass
    
    def batch_evaluate(self, args, data):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = RESPONSE_DICT.copy()
            res['prompt'] = prompt
            res['img_url'] = image
            res['lan'] = sample['lan']
            
            try:
                response = self.evaluate(prompt, image)
                res['response'] = response
            except Exception as e:
                print(f'Image{image} Error: {e}')
                res['response'] = 'Error'
            
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)