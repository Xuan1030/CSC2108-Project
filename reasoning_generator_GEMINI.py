import os, pickle, json, base64
from tqdm import tqdm

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

N = 5

# Universal prompts
user_prompt = f"Analyze {N} front, {N} middle, and {N} background feature words of the image given. Provide the answer in format: " + \
"'fr': 'feature_1', 'feature_2', ...; 'md': 'feature_1', 'feature_2', ...; 'bg': 'feature_1', 'feature_2', ..."


