"""
This is a script for calculating the tokens used.
"""
import sys
sys.path.append('..')

import math
import re
import os
from urllib import request
from io import BytesIO
import base64
import tiktoken

from PIL import Image
from utils import process_data
from tqdm import tqdm
from utils import dimensions

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Helper functions

def get_image_dims(image):
    # regex to check if image is a URL or base64 string
    url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    if re.match(url_regex, image):
        response = request.urlopen(image)
        image = Image.open(response)
        return image.size
    elif re.match(r'data:image\/\w+;base64', image):
        image = re.sub(r'data:image\/\w+;base64,', '', image)
        image = Image.open(BytesIO(base64.b64decode(image)))
        return image.size
    else:
        raise ValueError("Image must be a URL or base64 string.")

def calculate_image_token_cost(image, detail):
    # Constants
    LOW_DETAIL_COST = 85
    HIGH_DETAIL_COST_PER_TILE = 170
    ADDITIONAL_COST = 85

    if detail == 'low':
        # Low detail images have a fixed cost
        return LOW_DETAIL_COST
    elif detail == 'high':
        # Calculate token cost for high detail images
        width, height = get_image_dims(image)
        # Check if resizing is needed to fit within a 2048 x 2048 square
        if max(width, height) > 2048:
            # Resize the image to fit within a 2048 x 2048 square
            ratio = 2048 / max(width, height)
            width = int(width * ratio)
            height = int(height * ratio)

        # Further scale down to 768px on the shortest side
        if min(width, height) > 768:
            ratio = 768 / min(width, height)
            width = int(width * ratio)
            height = int(height * ratio)
        # Calculate the number of 512px squares
        num_squares = math.ceil(width / 512) * math.ceil(height / 512)

        # Calculate the total token cost
        total_cost = num_squares * HIGH_DETAIL_COST_PER_TILE + ADDITIONAL_COST

        return total_cost
    else:
        # Invalid detail_option
        raise ValueError("Invalid detail_option. Use 'low' or 'high'.")
    
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4-vision-preview" in model:
        num_tokens=0
        for message in messages:
            for key, value in message.items():
                if isinstance(value, list):
                    for item in value:
                        num_tokens += len(encoding.encode(item["type"]))
                        if item["type"] == "text":
                            num_tokens += len(encoding.encode(item["text"]))
                        elif item["type"] == "image_url":
                            num_tokens += calculate_image_token_cost(item["image_url"]["url"], item["image_url"]["detail"])
                elif isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def calculate_num_tokens(model, data):
    """
    Calculate the tokens used (vision).
    """
    tokens = 0
    for sample in tqdm(data):
        base64_image = encode_image(sample['img_url'])
        prompt = sample['prompt']
        message = [
            {
                'role': 'user',
                'content': [
                    {'type':'text', 'text':f'{prompt}'},
                    {'type':'image_url', 'image_url':{"url":f"data:image/jpeg;base64,{base64_image}", 'detail': 'high'}}
                ]
            }
        ]
        try:
            tokens += num_tokens_from_messages(message, model)
        except:
            tokens += 0
    print(f'Model Used: {model}, Token Used: {tokens}')
    
if __name__ == "__main__":
    for dim in dimensions:
        filepath = os.path.join('../data', dim)
        data = process_data(filepath)
        print(f'Current Dimension: {dim}.')
        calculate_num_tokens('gpt-4-vision-preview', data)