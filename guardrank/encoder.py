import pandas as pd
import jsonlines
from dataclasses import dataclass
from tqdm import tqdm

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import RobertaModel
from transformers import RobertaTokenizer

def load_data(filepath):
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for line in reader:
            data.append(line)
    return pd.DataFrame(data)

def get_data(data_path, language):
    data = load_data(data_path)
    prompt_text = (
        'Prompt: {prompt} Answer: {response}'
    )
    sample = []
    prompt = data['prompt']
    response = data['response']
    lan = data['lan']
    n = len(data)
    if language == 'en':
        for i in range(n):
            if lan[i] == 'en':
                sample.append({'text': prompt_text.format_map({'prompt': prompt[i], 'response': response[i]}), 'category': 0})
    elif language == 'zh':
        for i in range(n):
            if lan[i] == 'zh':
                sample.append({'text': prompt_text.format_map({'prompt': prompt[i], 'response': response[i]}), 'category': 0})
    else:
        for i in range(n):
            sample.append({'text': prompt_text.format_map({'prompt': prompt[i], 'response': response[i]}), 'category': 0})
    data = pd.DataFrame(sample)
    return data.to_dict('records')

class ClassDataset(torch.utils.data.Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data = self.transform(data_list)
    
    def transform(self, data_list):
        result_list = []
        for data in data_list:
            result = {}
            result['text'] = data['text']
            result['label'] = data['category']
            result_list.append(result)
        return result_list
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
@dataclass
class Collate:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_seq_len = max_length
        
    def __call__(self, batch):
        sources = []
        targets = []
        for example in batch:
            source = example['text']
            target = example['label']
            sources.append(source)
            targets.append(target)
        tokenized_sources = self.tokenizer(sources, max_length=self.max_seq_len, truncation=True)
        tokenized_sources['labels'] = targets
        
        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        
        for s, t, a in zip(tokenized_sources['input_ids'], tokenized_sources['labels'], tokenized_sources['attention_mask']):
            s = s[:self.max_seq_len]
            input_ids = s
            input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))
            attention_mask = a
            attention_mask += [0] * (self.max_seq_len - len(a))
            labels = t
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
        
        results = {
            'input_ids': torch.tensor(all_input_ids),
            'attention_mask': torch.tensor(all_attention_mask),
            'labels': torch.tensor(all_labels)
        }

        return results

class RobertaLargeClassifier(nn.Module):
    
    def __init__(self, model_name_or_path, num_labels=2, dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.roberta = RobertaModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask, *args, **kwargs):
        _, pooled_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_out)
        linear_output = self.linear(dropout_output)
        output = self.relu(linear_output)
        return output
    
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained(args.encoder_base)
    model = RobertaLargeClassifier(args.encoder_base)
    model.load_state_dict(torch.load(args.encoder_model))
    model = model.to(device)
    
    for evaluate_model in args.models:
        filepath = os.path.join(args.data_dir, f'hallucination_{evaluate_model}.jsonl')
        target_path = os.path.join(args.save_dir, 'hallucination', f'hallucination_{evaluate_model}.xlsx')
        if os.path.exists(target_path):
            print(f'File {target_path} has existed.')
            continue
        eval_df = load_data(filepath)
        eval_data = get_data(filepath, 'mix')
        eval_dataset = ClassDataset(eval_data)
        eval_dataloader = DataLoader(eval_dataset, 1, collate_fn=Collate(tokenizer, 256))
        
        n = len(eval_dataset)
        result_list = []
        
        model.eval()
        with torch.no_grad():
            with tqdm(eval_dataloader) as tbar:
                for eval_input in tbar:
                    mask = eval_input['attention_mask'].to(device)
                    input_ids = eval_input['input_ids'].squeeze(1).to(device)
                    output = model(input_ids, mask)
                    result_list.append(output.argmax(dim=1).item())
        
        results = pd.DataFrame(result_list, columns=['score'])
        results = pd.concat([eval_df, results], axis=1)
        results.to_excel(target_path, index=None)