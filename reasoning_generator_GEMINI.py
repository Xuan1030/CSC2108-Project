import os, json

from tqdm import tqdm
from pathlib import Path

import google.generativeai as genai


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

GOOGLE_PROJECT_ID = "gen-lang-client-0273375269"
GOOGLE_BUCKET_NAME = "csc2108_project"


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

N = 5

# Universal prompts
user_prompt = f"Analyze {N} front, {N} middle, and {N} background feature words of the image given. Provide the answer in format: " + \
"'fr': 'feature_1', 'feature_2', ...; 'md': 'feature_1', 'feature_2', ...; 'bg': 'feature_1', 'feature_2', ..."


def generate_batch_jsonl(bucket_name, image_folder_name):
  reqs = []
  for img_path in Path(image_folder_name).rglob('*.jpg'):
    img_path = str(img_path)
    
  return 

"gs://csc2108_project/original_images/American Samoa/canvas_1629271395.jpg"