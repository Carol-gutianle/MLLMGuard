"""
This is a script for calculating the tokens used for gemini.
"""
import sys
sys.path.append('..')

import google.generativeai as genai
from utils import process_data
from tqdm import tqdm
from rich import print
from utils import dimensions

# please replace None with API_KEY
API_KEY = None

def calculate_num_tokens(data):
    genai.configure(api_key=API_KEY, transport="rest")
    generation_config = {
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 128
    }
    model = genai.GenerativeModel(
        model_name = 'gemini-pro-vision',
        generation_config = generation_config
    )
    cnt = 0
    # 遍历样本
    for sample in tqdm(data):
        text_prompt = sample['prompt']
        cnt += model.count_tokens(text_prompt).total_tokens
    return cnt
    
def batch_calculate():
    final_dimensions = ['privacy', 'bias', 'toxicity', 'truthfulness', 'legality']
    results = {}
    for dim in dimensions:
        data = process_data(f'../data/{dim}')
        cnt = calculate_num_tokens(data)
        # completion_tokens = prompt_tokens + 128 * prompt_tokens
        # total_cost = 0.0025 * num_images + (prompt_tokens / 1000) * 0.000125 + (128 * num_samples / 1000) * 0.000375
        # reference: 
        results[dim] = {'prompt_tokens': cnt, 'completion_tokens': cnt + 128 * len(data), 'total_cost': 0.0025 * len(data) + (cnt / 1000) * 0.000125 + (128 * len(data) / 1000) * 0.000375}
    for dim in final_dimensions:
        curr_result = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0.0}.copy()
        if dim == 'truthfulness':
            curr_result['prompt_tokens'] = results['non-existent']['prompt_tokens'] + results['position-swapping']['prompt_tokens'] + results['noise-injection']['prompt_tokens']
            curr_result['completion_tokens'] = results['non-existent']['completion_tokens'] + results['position-swapping']['completion_tokens'] + results['noise-injection']['completion_tokens']
            curr_result['total_cost'] = results['non-existent']['total_cost'] + results['position-swapping']['total_cost'] + results['noise-injection']['total_cost']
        else:
            curr_result['prompt_tokens'] = results[dim]['prompt_tokens']
            curr_result['completion_tokens'] = results[dim]['completion_tokens']
            curr_result['total_cost'] = results[dim]['total_cost']
        print(f'''[red]dimension: {dim.ljust(6)}[/red]\t[blue]prompt_tokens: {curr_result['prompt_tokens']}[/blue]\t[green]completion_tokens:{curr_result['completion_tokens']}[/green]\t[brown]total_cost:{round(curr_result['total_cost'], 2)}[/brown]\n''')
    
if __name__ == "__main__":
    batch_calculate()