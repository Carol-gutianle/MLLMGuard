import pandas as pd
import os
import jsonlines
from tqdm import tqdm

# global settings
dimensions = ['privacy', 'bias', 'toxicity', 'hallucination', 'noise-injection', 'position-swapping', 'legality']


# response的格式
RESPONSE_DICT = {
    'prompt': '',
    'img_url': '',
    'response': '',
    'lan': ''
}

# 输入的格式
INPUT_DICT = {
    'index': 0,
    'img_url': '',
    'prompt': '',
    'lan': '',
    'type': 'free-text' # or choice
}

def process_data(data_path):
    """
    输入是 csv + img
    """
    data_list = []
    img_base = os.path.join(data_path, 'imgs')
    data = pd.read_csv(os.path.join(data_path, 'prompt.csv'))
    n = data.shape[0]
    cnt = 0
    if 'sequential' in data_path:
        for i in range(n):
            for j in range(2):
                new_result = INPUT_DICT.copy()
                new_result['index'] = cnt
                new_result['img_url'] = os.path.join(img_base, data.iat[i, j])
                new_result['prompt'] = data.iat[i, 2]
                new_result['lan'] = data.iat[i, 5]
                new_result['type'] = 'choice'
                data_list.append(new_result)
                cnt += 1
                
    elif 'non-existent' in data_path:
        for i in range(n):
            for j in range(2):
                new_result = INPUT_DICT.copy()
                new_result['index'] = cnt
                new_result['img_url'] = os.path.join(img_base, data.iat[i, 0])
                if j == 0:
                    new_result['prompt'] = data.iat[i, 1]
                    new_result['lan'] = data.iat[i, 5]
                else:
                    new_result['prompt'] = f'{data.iat[i,1]}(Please answer me with options) {data.iat[i, 2]}'
                    new_result['type'] = f'choice'
                    new_result['lan'] = data.iat[i, 5]
                data_list.append(new_result)
                cnt += 1
                
    elif 'noise-consistency' in data_path:
        for i in range(n):
            for j in range(2):
                new_result = INPUT_DICT.copy()
                new_result['index'] = cnt
                new_result['prompt'] = data.iat[i, 1]
                if j == 0:
                    new_result['img_url'] = os.path.join(img_base, data.iat[i, 0])
                else:
                    base_name = os.path.basename(data.iat[i, 0])
                    name, ext = os.path.splitext(base_name)
                    new_result['img_url'] = os.path.join(img_base, f'{name}_noise{ext}')
                    new_result['type'] = 'add_noise'
                new_result['lan'] = data.iat[i, 5]
                cnt += 1
                data_list.append(new_result)   
                                                     
    else:
        for i in range(n):
            new_result = INPUT_DICT.copy()
            new_result['index'] = i
            new_result['img_url'] = os.path.join(img_base, data.iat[i, 0])
            new_result['prompt'] = data.iat[i, 1]
            new_result['lan'] = data.iat[i, 4]
            data_list.append(new_result)
             
    return data_list

def load_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in tqdm(reader, desc="Loading data..."):
            data.append(line)
        return data
    
def save_data(data, save_path):
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(data)