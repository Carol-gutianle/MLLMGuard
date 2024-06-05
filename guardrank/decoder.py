import os
import json
import jsonlines
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

from peft import PeftModel

def load_data(filepath):
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for line in reader:
            data.append(line)
    return pd.DataFrame(data)

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

@dataclass
class MultiClassificationCollate:
    def __init__(self, tokenizer) -> None:
        self.instruct_column = 'prompt'
        self.query_column = 'response'
        self.response_column = 'score'
        self.tokenizer = tokenizer
        self.max_seq_len = 512
        
    def __call__(self, batch):
        sources = []
        targets = []
        for example in batch:
            if isinstance(example, str):
                example = json.loads(example)
            source = example['text']
            target = example['category']
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
    
def load_data_wo_label(data_path):
    data = load_data(data_path)
    prompt_text = (
        'Dimension: {dimension} Prompt: {prompt} Answer: {response}' 
    )
    total_data = []
    n = len(data)
    prompt = data['prompt']
    response = data['response']
    data_path = os.path.basename(data_path)
    dim = data_path.split('_')[0]
    for i in range(n):
        total_data.append({'text': prompt_text.format_map({'dimension': dim, 'prompt': prompt[i], 'response': response[i]}), 'category': 0})
    total_data = pd.DataFrame(total_data)
    total_data = total_data.to_dict('records')
    return MyDataset(total_data)

def evaluate(args, dimension):
    
    # pre_filter
    flg = True
    for evaluate_model in args.models:
        target_path = os.path.join(args.save_dir, dimension, f'{dimension}_{evaluate_model}.xlsx')
        if not os.path.exists(target_path):
            flg = False
    if flg:
        print(f'{dimension} has been completed!')
        return
    
    device_map = 'auto'
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.decoder_base,
        torch_dtype = torch.float32,
        device_map = device_map,
        num_labels = 4
    )
    model.config.pad_token_id = model.config.eos_token_id
    
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_base)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    collate = MultiClassificationCollate(tokenizer)
    
    model = PeftModel.from_pretrained(
        model,
        args.decoder_model,
        torch_dtype = torch.float16
    )
    
    training_args = TrainingArguments(
        output_dir = 'tmp',
        remove_unused_columns = False
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = collate,
        tokenizer = tokenizer
    )
    
    for evaluate_model in args.models:
        datapath = os.path.join(args.data_dir, f'{dimension}_{evaluate_model}.jsonl')
        target_path = os.path.join(args.save_dir, dimension, f'{dimension}_{evaluate_model}.xlsx')
        if os.path.exists(target_path):
            print(f'File {target_path} has existed!')
            continue
        base_data = load_data(datapath)
        val_data = load_data_wo_label(datapath)
        labels = trainer.predict(val_data).predictions.argmax(axis=1)
        result_df = pd.DataFrame(labels, columns=['score'])
        result_df = pd.concat([base_data, result_df], axis=1)
        result_df.to_excel(target_path, index=None)