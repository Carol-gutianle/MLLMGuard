from multiprocessing.managers import ValueProxy
from models.base import Mllm
from pathlib import Path
import mimetypes
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class Gemini(Mllm):
  def __init__(self, model_name_or_path, *args, **kwargs) -> None:
    super().__init__(model_name_or_path, *args, **kwargs)
    generation_config = {
      "temperature": 0.0,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 128,
    }
    genai.configure(api_key=model_name_or_path, transport="rest")
    self.model = genai.GenerativeModel(
      model_name = "gemini-pro-vision",
      generation_config = generation_config,
    )
    
  def __call__(self, prompt, filepath):
    status = 0
    while status != 1:
      try:
        mime_type, _ = mimetypes.guess_type(filepath)
        image_parts = [
          {
            'mime_type': mime_type,
            'data': Path(filepath).read_bytes()
          }
        ]
        prompt_parts = [
          prompt,
          image_parts[0],
          " "
        ]
        response = self.model.generate_content(prompt_parts)
        status = 1
        time.sleep(2)
      except Exception as e:
        print(e)
        time.sleep(2)
        raise ValueError(1)
    return response.text
  
  def evaluate(self, prompt, filepath):
    try:
      output = self.__call__(prompt, filepath)
    except:
      output = 'err'
    return output