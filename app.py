#@title Prepare the Concepts Library to be used
import json
import shutil
import sqlite3
import subprocess
import sys
sys.path.append('src/blip')
sys.path.append('src/clip')
import clip
import hashlib
import math
import numpy as np
import pickle
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import requests
import wget
import gradio as grad, random, re
import gradio as gr
import torch
import os
import utils
import html
import re
import base64
import subprocess
import argparse
import logging
import streamlit as st
import pandas as pd
import datasets
import yaml
import textwrap
import tornado
import time
from torch import autocast
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
from huggingface_hub import HfApi
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionImg2ImgPipeline 
from PIL import Image
from datasets import load_dataset
from share_btn import community_icon_html, loading_icon_html, share_js
from io import BytesIO
from models.blip import blip_decoder
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from flask import Flask, request, jsonify, g
from flask_expects_json import expects_json
from flask_cors import CORS
from huggingface_hub import Repository
from flask_apscheduler import APScheduler
from jsonschema import ValidationError
from os import mkdir
from os.path import isdir
from colorthief import ColorThief
from data_measurements.dataset_statistics import DatasetStatisticsCacheClass as dmt_cls
from utils import dataset_utils
from utils import streamlit_utils as st_utils
from dataclasses import asdict
#from torch import autocast
#from diffusers import StableDiffusionPipeline
#from io import BytesIO
#import base64
#import torch

from share_btn import community_icon_html, loading_icon_html, share_js

subprocess.run(["make", "build-all"], shell=False)
img_to_text = gr.Blocks.load(name="spaces/pharma/CLIP-Interrogator")
stable_diffusion = gr.Blocks.load(name="spaces/stabilityai/stable-diffusion")

is_colab = utils.is_google_colab()

MODE = os.environ.get('FLASK_ENV', 'production')
IS_DEV = MODE == 'development'
app = Flask(__name__, static_url_path='/static')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

schema = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "minProperties": 2,
                "maxProperties": 2,
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "maxItems": 5,
                        "minItems": 5
                    },
                    "imgURL": {"type": "string"}}
            }
        }
    },
    "minProperties": 2,
    "maxProperties": 2
}

CORS(app)

DB_FILE = Path("./data.db")
TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN')
repo = Repository(
    local_dir="data",
    repo_type="dataset",
    clone_from="huggingface-projects/color-palettes-sd",
    use_auth_token=TOKEN
)
repo.git_pull()
# copy db on db to local path
shutil.copyfile("./data/data.db", DB_FILE)

db = sqlite3.connect(DB_FILE)
try:
    data = db.execute("SELECT * FROM palettes").fetchall()
    if IS_DEV:
        print(f"Loaded {len(data)} palettes from local db")
    db.close()
except sqlite3.OperationalError:
    db.execute(
        'CREATE TABLE palettes (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, data json, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)')
    db.commit()

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

if torch.cuda.is_available():
  torchfloat = torch.float16
else:
  torchfloat = torch.float32


class Model:
    def __init__(self, name, path, prefix):
        self.name = name
        self.path = path
        self.prefix = prefix

models = [
     Model("Custom model", "", ""),
     Model("Arcane", "nitrosocke/Arcane-Diffusion", "arcane style "),
     Model("Archer", "nitrosocke/archer-diffusion", "archer style "),
     Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
     Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
     Model("Modern Disney", "nitrosocke/modern-disney-diffusion", "modern disney style "),
     Model("Classic Disney", "nitrosocke/classic-anim-diffusion", ""),
     Model("Waifu", "hakurei/waifu-diffusion", ""),
     Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
     Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
     Model("Robo Diffusion", "nousr/robo-diffusion", ""),
     Model("Cyberpunk Anime", "DGSpitzer/Cyberpunk-Anime-Diffusion", "dgs illustration style "),
     Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy")
]

last_mode = "txt2img"
current_model = models[1]
current_model_path = current_model.path

models = [
  "DGSpitzer/Cyberpunk-Anime-Diffusion"
]

prompt_prefixes = {
  models[0]: "dgs illustration style "
}

current_model = models[0]

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

#If you are running this code locally, you need to either do a 'huggingface-cli login` or paste your User Access Token from here https://huggingface.co/settings/tokens into the use_auth_token field below. 
#pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, revision="fp16", torch_dtype=torch.float16)
#pipe = pipe.to(device)
#torch.backends.cudnn.benchmark = True

#auth_token = os.environ.get("test") or True
#pipe = StableDiffusionPipeline.from_pretrained(current_model, use_auth_token=auth_token, torch_dtype=torchfloat, revision="fp16")
#model_id = "hakurei/waifu-diffusion"
pipe = StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion", torch_type=torch.float16, revision="fp16")
pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torchfloat, revision="fp16")
gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True, revision="fp16", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(current_model.path, torch_dtype=torch.float16)
# pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(current_model.path, torch_dtype=torch.float16)

with open("ideas.txt", "r") as f:
    line = f.readlines()

if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  # pipe_i2i = pipe_i2i.to("cuda")
else:
  pipe = pipe.to("cpu")
  
device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

#torch.backends.cudnn.benchmark = True
num_samples = 2

is_gpu_busy = False

def on_model_change(model):

    global current_model
    global pipe
    if model != current_model:
        current_model = model
        pipe = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torchfloat)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

def inference(prompt, guidance, steps):

    promptPrev = prompt
    prompt = prompt_prefixes[current_model] + prompt
    image = pipe(prompt, num_inference_steps=int(steps), guidance_scale=guidance, width=512, height=512).images[0]
    return image, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(placeholder=promptPrev)

def inference_example(prompt, guidance, steps):

    prompt = prompt_prefixes[current_model] + prompt
    image = pipe(prompt, num_inference_steps=int(steps), guidance_scale=guidance, width=512, height=512).images[0]
    return image

def infer(prompt):
    images = pipe([prompt] * num_samples, guidance_scale=7.5)["sample"]
    global is_gpu_busy
    samples = 4
    steps = 50
    scale = 7.5
        
    #generator = torch.Generator(device=device).manual_seed(seed)
    #print("Is GPU busy? ", is_gpu_busy)
    images = []
    #if(not is_gpu_busy):
    #    is_gpu_busy = True
    #    images_list = pipe(
    #        [prompt] * samples,
    #        num_inference_steps=steps,
    #        guidance_scale=scale,
            #generator=generator,
    #    )
    #    is_gpu_busy = False
    #    for i, image in enumerate(images_list["sample"]):
     #           images.append(image)
    #else:
    url = os.getenv('JAX_BACKEND_URL')
    payload = {'prompt': prompt}
    images_request = requests.post(url, json = payload)
    for image in images_request.json()["images"]:
        image_b64 = (f"data:image/jpeg;base64,{image}")
        images.append(image_b64)
        
    return images

def generate(starting_text):
    seed = random.randint(100, 1000000)
    set_seed(seed)

    if starting_text == "":
        starting_text: str = line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

    response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            response_list.append(resp+'\n')

    response_end = "\n".join(response_list)
    response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
    response_end = response_end.replace("<", "").replace(">", "")

    if response_end != "":
        return response_end

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  
  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  i = 1
  while(num_added_tokens == 0):
    print(f"The tokenizer already contains the token {token}.")
    token = f"{token[:-1]}-{i}>"
    print(f"Attempting to add the token {token}.")
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token

print("Setting up the public library")
for model in models_list:
  model_content = {}
  model_id = model.modelId
  model_content["id"] = model_id
  embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
  os.makedirs(model_id,exist_ok = True)
  if not os.path.exists(f"{model_id}/learned_embeds.bin"):
    try:
      wget.download(embeds_url, out=model_id)
    except:
      continue
  token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
  response = requests.get(token_identifier)
  token_name = response.text
  
  concept_type = f"https://huggingface.co/{model_id}/raw/main/type_of_concept.txt"
  response = requests.get(concept_type)
  concept_name = response.text
  model_content["concept_type"] = concept_name
  images = []
  for i in range(4):
    url = f"https://huggingface.co/{model_id}/resolve/main/concept_images/{i}.jpeg"
    image_download = requests.get(url)
    url_code = image_download.status_code
    if(url_code == 200):
      file = open(f"{model_id}/{i}.jpeg", "wb") ## Creates the file for image
      file.write(image_download.content) ## Saves file content
      file.close()
      images.append(f"{model_id}/{i}.jpeg")
  model_content["images"] = images
  #if token cannot be loaded, skip it
  try:
    learned_token = load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", pipe.text_encoder, pipe.tokenizer, token_name)
  except: 
    continue
  model_content["token"] = learned_token
  models.append(model_content)
  
#@title Run the app to navigate around [the Library](https://huggingface.co/sd-concepts-library)
#@markdown Click the `Running on public URL:` result to run the Gradio app

SELECT_LABEL = "Select concept"
def assembleHTML(model):
  html_gallery = ''
  html_gallery = html_gallery+'''
  <div class="flex gr-gap gr-form-gap row gap-4 w-full flex-wrap" id="main_row">
  '''
  cap = 0
  for model in models:
    html_gallery = html_gallery+f'''
    <div class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200 gr-panel">
      <div class="output-markdown gr-prose" style="max-width: 100%;">
        <h3>
          <a href="https://huggingface.co/{model["id"]}" target="_blank">
            <code>{html.escape(model["token"])}</code>
          </a>
        </h3>
      </div>
      <div id="gallery" class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200">
        <div class="wrap svelte-17ttdjv opacity-0"></div>
        <div class="absolute left-0 top-0 py-1 px-2 rounded-br-lg shadow-sm text-xs text-gray-500 flex items-center pointer-events-none bg-white z-20 border-b border-r border-gray-100 dark:bg-gray-900">
          <span class="mr-2 h-[12px] w-[12px] opacity-80">
            <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </span> {model["concept_type"]}
        </div>
        <div class="overflow-y-auto h-full p-2" style="position: relative;">
          <div class="grid gap-2 grid-cols-2 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-2 svelte-1g9btlg pt-6">
        '''
    for image in model["images"]:
                html_gallery = html_gallery + f'''    
                <button class="gallery-item svelte-1g9btlg">
                  <img alt="" loading="lazy" class="h-full w-full overflow-hidden object-contain" src="file/{image}">
                </button>
                '''
    html_gallery = html_gallery+'''
              </div>
              <iframe style="display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; border: 0; opacity: 0; pointer-events: none; z-index: -1;" aria-hidden="true" tabindex="-1" src="about:blank"></iframe>
            </div>
          </div>
        </div>
        '''
    cap += 1
    if(cap == 99):
      break  
  html_gallery = html_gallery+'''
  </div>
  '''
  return html_gallery
  
def title_block(title, id):
  return gr.Markdown(f"### [`{title}`](https://huggingface.co/{id})")

def image_block(image_list, concept_type):
  return gr.Gallery(
          label=concept_type, value=image_list, elem_id="gallery"
          ).style(grid=[2], height="auto")

def checkbox_block():
  checkbox = gr.Checkbox(label=SELECT_LABEL).style(container=False)
  return checkbox

def infer(text):
  with autocast("cuda"):
    images_list = pipe(
              [text]*2,
              num_inference_steps=50,
              guidance_scale=7.5
  )
  output_images = []
  for i, image in enumerate(images_list["sample"]):
    output_images.append(image)
  return output_images, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

# idetnical to `infer` function without gradio state updates for share btn
def infer_examples(text):
  with autocast("cuda"):
    images_list = pipe(
              [text]*2,
              num_inference_steps=50,
              guidance_scale=7.5
  )
  output_images = []
  for i, image in enumerate(images_list["sample"]):
    output_images.append(image)
  return output_images

txt = grad.Textbox(lines=1, label="Initial Text", placeholder="English Text here")
out = grad.Textbox(lines=4, label="Generated Prompts")

examples = []
for x in range(8):
    examples.append(line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize())
    
def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None

  if img is not None:
    return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator)
  else:
    return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator)

def txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator=None):

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        pipe = StableDiffusionPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16)
        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      # num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
def img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height, generator=None):

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model_path, torch_dtype=torch.float16)
        
        if torch.cuda.is_available():
              pipe = pipe.to("cuda")
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        # num_images_per_prompt=n_images,
        init_image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        width = width,
        height = height,
        generator = generator)    

def get_images(prompt):
    gallery_dir = stable_diffusion(prompt, fn_index=2)
    sd_output = [os.path.join(gallery_dir, image) for image in os.listdir(gallery_dir)]
    return sd_output, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def get_prompts(uploaded_image):
    return img_to_text(uploaded_image, fn_index=1)[0]

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def update_repository():
    repo.git_pull()
    # copy db on db to local path
    shutil.copyfile(DB_FILE, "./data/data.db")

    with sqlite3.connect("./data/data.db") as db:
        db.row_factory = sqlite3.Row
        palettes = db.execute("SELECT * FROM palettes").fetchall()
        data = [{'id': row['id'], 'data': json.loads(
            row['data']), 'created_at': row['created_at']} for row in palettes]

    with open('./data/data.json', 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    print("Updating repository")
    subprocess.Popen(
        "git add . && git commit --amend -m 'update' && git push --force", cwd="./data", shell=True)
    repo.push_to_hub(blocking=False)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/force_push')
def push():
    if (request.headers['token'] == TOKEN):
        update_repository()
        return jsonify({'success': True})
    else:
        return "Error", 401

def getAllData():
    palettes = get_db().execute("SELECT * FROM palettes").fetchall()
    data = [{'id': row['id'], 'data': json.loads(
            row['data']), 'created_at': row['created_at']} for row in palettes]
    return data

@app.route('/data')
def getdata():
    return jsonify(getAllData())

@app.route('/new_palette', methods=['POST'])
@expects_json(schema)
def create():
    data = g.data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO palettes(data) VALUES (?)", [json.dumps(data)])
    db.commit()
    return jsonify(getAllData())

@app.errorhandler(400)
def bad_request(error):
    if isinstance(error.description, ValidationError):
        original_error = error.description
        return jsonify({'error': original_error.message}), 400
    return error

if __name__ == '__main__':
    if not IS_DEV:
        print("Starting scheduler -- Running Production")
        scheduler = APScheduler()
        scheduler.add_job(id='Update Dataset Repository',
                          func=update_repository, trigger='interval', hours=1)
        scheduler.start()
    else:
        print("Not Starting scheduler -- Running Development")
    app.run(host='0.0.0.0',  port=int(
        os.environ.get('PORT', 7860)), debug=True, use_reloader=IS_DEV)        

title = "Stable Diffusion Prompt Generator"
description = 'This is a demo of the model series: "MagicPrompt", in this case, aimed at: "Stable Diffusion". To use it, simply submit your text or click on one of the examples. To learn more about the model, [click here](https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion).<br>'

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               examples=examples,
               title=title,
               description=description,
               article='',
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)

css = """
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
a {text-decoration-line: underline;}
  <style>
  .finetuned-diffusion-div {
      text-align: center;
      max-width: 700px;
      margin: 0 auto;
    }
    .finetuned-diffusion-div div {
      display: inline-flex;
      align-items: center;
      gap: 0.8rem;
      font-size: 1.75rem;
    }
    .finetuned-diffusion-div div h1 {
      font-weight: 900;
      margin-bottom: 7px;
    }
    .finetuned-diffusion-div p {
      margin-bottom: 10px;
      font-size: 94%;
    }
    .finetuned-diffusion-div p a {
      text-decoration: underline;
    }
    .tabs {
      margin-top: 0px;
      margin-bottom: 0px;
    }
    #gallery {
      min-height: 20rem;
    }
  </style>
.gradio-container {font-family: 'IBM Plex Sans', sans-serif}
#top_title{margin-bottom: .5em}
#top_title h2{margin-bottom: 0; text-align: center}
/*#main_row{flex-wrap: wrap; gap: 1em; max-height: 550px; overflow-y: scroll; flex-direction: row}*/
#component-3{height: 760px; overflow: auto}
#component-9{position: sticky;top: 0;align-self: flex-start;}
@media (min-width: 768px){#main_row > div{flex: 1 1 32%; margin-left: 0 !important}}
.gr-prose code::before, .gr-prose code::after {content: "" !important}
::-webkit-scrollbar {width: 10px}
::-webkit-scrollbar-track {background: #f1f1f1}
::-webkit-scrollbar-thumb {background: #888}
::-webkit-scrollbar-thumb:hover {background: #555}
.gr-button {white-space: nowrap}
.gr-button:focus {
  border-color: rgb(147 197 253 / var(--tw-border-opacity));
  outline: none;
  box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
  --tw-border-opacity: 1;
  --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
  --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
  --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
  --tw-ring-opacity: .5;
}
#prompt_input{flex: 1 3 auto; width: auto !important;}
#prompt_area{margin-bottom: .75em}
#prompt_area > div:first-child{flex: 1 3 auto}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
a {text-decoration-line: underline; font-weight: 600;}
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
       transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
}
#share-btn * {
     all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
        .gradio-container
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 700px
        }
"""
examples = ["a <cat-toy> in <madhubani-art> style", "a <line-art> style mecha robot", "a piano being played by <bonzi>", "Candid photo of <cheburashka>, high resolution photo, trending on artstation, interior design"]

block = gr.Blocks(css=css)

examples = [
    [
        'Goku'
    ],
    [
        'Mikasa Ackerman'
    ],
    [
        'Saber'
    ],
]

with block as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <img src="data:image/png;base64,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" />
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Waifu Diffusion Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                waifu-diffusion is a latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning
              </p>
            </div>
        """)

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  DGS Diffusion Space
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
               Demo for Cyberpunk Anime Diffusion. Based of the projects by anzorq and fffiloni <a href="https://twitter.com/hahahahohohe">
              </p>
            </div>
        """
    )
    gr.Markdown('''
      👇 Buy me a coffee if you like ♥ this project~ 👇 Running this server costs me $100 per week, any help would be much appreciated!
      [![Buy me a coffee](https://badgen.net/badge/icon/Buy%20Me%20A%20Coffee?icon=buymeacoffee&label)](https://www.buymeacoffee.com/dgspitzer)
    ''')
    with gr.Row():
        
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=models, value=models[0])
            prompt = gr.Textbox(label="Prompt", placeholder="{} is added automatically".format(prompt_prefixes[current_model]), elem_id="input-prompt")
            guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
            steps = gr.Slider(label="Steps", value=27, maximum=100, minimum=2)
            run = gr.Button(value="Run")
            gr.Markdown(f"Running on: {device}")
        with gr.Column():
            image_out = gr.Image(height=512, type="filepath", elem_id="output-img")

    with gr.Column(elem_id="col-container"):
        with gr.Group(elem_id="share-btn-container"):
          community_icon = gr.HTML(community_icon_html, visible=False)
          loading_icon = gr.HTML(loading_icon_html, visible=False)
          share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
          
    model.change(on_model_change, inputs=model, outputs=[])
    run.click(inference, inputs=[prompt, guidance, steps], outputs=[image_out, share_button, community_icon, loading_icon, prompt])
    
    share_button.click(None, [], [], _js=share_js)
       
    gr.Examples([
        ["portrait of anime girl", 7.5, 27],
        ["a beautiful perfect face girl, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece by studio ghibli, 8k, sharp high quality anime, artstation", 7.5, 27],
        ["cyberpunk city landscape with fancy car", 7.5, 27],
        ["portrait of liu yifei girl, soldier working in a cyberpunk city, cleavage, intricate, 8k, highly detailed, digital painting, intense, sharp focus", 7.5, 27],
        ["portrait of a muscular beard male in dgs illustration style, half-body, holding robot arms, strong chest", 7.5, 27],
    ], [prompt, guidance, steps], image_out, inference_example, cache_examples=torch.cuda.is_available())
    gr.Markdown('''
      Models and Space by [@DGSpitzer](https://huggingface.co/DGSpitzer)❤️<br>
      [![Twitter Follow](https://img.shields.io/twitter/follow/DGSpitzer?label=%40DGSpitzer&style=social)](https://twitter.com/DGSpitzer)
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=dgspitzer_DGS_Diffusion_Space)
      
      ![Model Views](https://visitor-badge.glitch.me/badge?page_id=Cyberpunk_Anime_Diffusion)
      
    ''')

with gr.Blocks(css=css) as demo:
  state = gr.Variable({
        'selected': -1
  })
  state = {}
  def update_state(i):
        global checkbox_states
        if(checkbox_states[i]):
          checkbox_states[i] = False
          state[i] = False
        else:
          state[i] = True
          checkbox_states[i] = True   
  gr.HTML('''
  <div style="text-align: center; max-width: 720px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Conceptualizer
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Navigate through community created concepts and styles via Stable Diffusion Textual Inversion and pick yours for inference.
                To train your own concepts and contribute to the library <a style="text-decoration: underline" href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb">check out this notebook</a>.
              </p>
            </div> ''')
     with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
               
        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="generated_id").style(
            grid=[2], height="auto"
        )
        
        ex = gr.Examples(examples=examples, fn=infer, inputs=[text], outputs=gallery, cache_examples=True)
        ex.dataset.headers = [""]
        
        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)

    gr.HTML(
            """
                <div class="footer">
                    <p>Stable Diffusion model fine-tuned on 56K anime image board images by <a href="https://huggingface.co/hakurei" style="text-decoration: underline;" target="_blank">hakurei</a>
                    </p>
                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """
         )
       with gr.Row():
        with gr.Column():
          gr.Markdown(f"### Navigate the top 100 Textual-Inversion community trained concepts. Use 600+ from [The Library](https://huggingface.co/sd-concepts-library)")
          with gr.Row():
                  image_blocks = []
                  #for i, model in enumerate(models):
                  with gr.Box().style(border=None):
                    gr.HTML(assembleHTML(models))
                      #title_block(model["token"], model["id"])
                      #image_blocks.append(image_block(model["images"], model["concept_type"]))
        with gr.Column():
          with gr.Box():
                  with gr.Row(elem_id="prompt_area").style(mobile_collapse=False, equal_height=True):
                      text = gr.Textbox(
                          label="Enter your prompt", placeholder="Enter your prompt", show_label=False, max_lines=1, elem_id="prompt_input"
                      ).style(
                          border=(True, False, True, True),
                          rounded=(True, False, False, True),
                          container=False,
                          full_width=False,
                      )
                      btn = gr.Button("Run",elem_id="run_btn").style(
                          margin=False,
                          rounded=(False, True, True, False),
                          full_width=False,
                      )  
                  with gr.Row().style():
                      infer_outputs = gr.Gallery(show_label=False, elem_id="generated-gallery").style(grid=[2], height="512px")
                  with gr.Row():
                    gr.HTML("<p style=\"font-size: 95%;margin-top: .75em\">Prompting may not work as you are used to. <code>objects</code> may need the concept added at the end, <code>styles</code> may work better at the beginning. You can navigate on <a href='https://lexica.art'>lexica.art</a> to get inspired on prompts</p>")
                  with gr.Row():
                    gr.Examples(examples=examples, fn=infer_examples, inputs=[text], outputs=infer_outputs, cache_examples=True)
          with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
  checkbox_states = {}
  inputs = [text]
  btn.click(
        infer,
        inputs=inputs,
        outputs=[infer_outputs, community_icon, loading_icon, share_button]
    )
  share_button.click(
      None,
      [],
      [],
      _js=share_js,)       
       with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Finetuned Diffusion</h1>
              </div>
              <p>
               Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: <br>
               <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>, <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spiderverse</a>, <a href="https://huggingface.co/nitrosocke/modern-disney-diffusion">Modern Disney</a>, <a href="https://huggingface.co/hakurei/waifu-diffusion">Waifu</a>, <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokemon</a>, <a href="https://huggingface.co/yuk/fuyuko-waifu-diffusion">Fuyuko Waifu</a>, <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony</a>, <a href="https://huggingface.co/sd-dreambooth-library/herge-style">Hergé (Tintin)</a>, <a href="https://huggingface.co/nousr/robo-diffusion">Robo</a>, <a href="https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion">Cyberpunk Anime</a> + any other custom Diffusers 🧨 SD model hosted on HuggingFace 🤗.
              </p>
              <p>Don't want to wait in queue? <a href="https://colab.research.google.com/gist/qunash/42112fb104509c24fd3aa6d1c11dd6e0/copy-of-fine-tuned-diffusion-gradio.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a></p>
               Running on <b>{device}</b>
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Group():
            model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
            custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", visible=False, interactive=True)
            
            with gr.Row():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="Enter prompt. Style applied automatically").style(container=False)
              generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))


            image_out = gr.Image(height=512)
            # gallery = gr.Gallery(
            #     label="Generated images", show_label=False, elem_id="gallery"
            # ).style(grid=[1], height="auto")

        with gr.Tab("Options"):
          with gr.Group():
            neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

            # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

            with gr.Row():
              guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
              steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=100, step=1)

            with gr.Row():
              width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
              height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

            seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

        with gr.Tab("Image to image"):
            with gr.Group():
              image = gr.Image(label="Image", height=256, tool="editor", type="pil")
              strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    model_name.change(lambda x: gr.update(visible = x == models[0].name), inputs=model_name, outputs=custom_model_path)
    custom_model_path.change(custom_model_changed, inputs=custom_model_path)
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    inputs = [model_name, prompt, guidance, steps, width, height, seed, image, strength, neg_prompt]
    prompt.submit(inference, inputs=inputs, outputs=image_out)
    generate.click(inference, inputs=inputs, outputs=image_out)

    ex = gr.Examples([
        [models[1].name, "jason bateman disassembling the demon core", 7.5, 50],
        [models[4].name, "portrait of dwayne johnson", 7.0, 75],
        [models[5].name, "portrait of a beautiful alyx vance half life", 10, 50],
        [models[6].name, "Aloy from Horizon: Zero Dawn, half body portrait, smooth, detailed armor, beautiful face, illustration", 7.0, 45],
        [models[5].name, "fantasy portrait painting, digital art", 4.0, 30],
    ], [model_name, prompt, guidance, steps, seed], image_out, inference, cache_examples=not is_colab and torch.cuda.is_available())
    # ex.dataset.headers = [""]

    gr.Markdown('''
      Models by [@nitrosocke](https://huggingface.co/nitrosocke), [@Helixngc7293](https://twitter.com/DGSpitzer) and others. ❤️<br>
      Space by: [![Twitter Follow](https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social)](https://twitter.com/hahahahohohe)
  
      ![visitors](https://visitor-badge.glitch.me/badge?page_id=anzorq.finetuned_diffusion)
    ''')
    block = gr.Blocks(css=css)

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
#        4,
#        45,
#        7.5,
#        1024,
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        'A mecha robot in a favela in expressionist style',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        'an insect robot preparing a delicious meal',
#        4,
#        45,
#        7,
#        1024,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
#        4,
#        45,
#        7,
#        1024,
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Stable Diffusion is a state of the art text-to-image model that generates
                images from text.<br>For faster generation and API
                access you can try
                <a
                  href="http://beta.dreamstudio.ai/"
                  style="text-decoration: underline;"
                  target="_blank"
                  >DreamStudio Beta</a
                >
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

        with gr.Row(elem_id="advanced-options"):
            gr.Markdown("Advanced settings are temporarily unavailable")
            samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(examples=examples, fn=infer, inputs=text, outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        ex.dataset.headers = [""]

        text.submit(infer, inputs=text, outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=text, outputs=[gallery], postprocess=False)
        
        advanced_button.click(
            None,
            [],
            text,
            _js="""
            () => {
                const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
                options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
            }""",
        )
        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )
        gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/CompVis" style="text-decoration: underline;" target="_blank">CompVis</a> and <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">Stability AI</a> - backend running JAX on TPUs due to generous support of <a href="https://sites.research.google/trc/about/" style="text-decoration: underline;" target="_blank">Google TRC program</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
                </div>
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
           """)   

     with gr.Row():
       with gr.Column():
           input_img = gr.Image(type="filepath", elem_id="input-img")
           with gr.Row():
             see_prompts = gr.Button("Feed in your image!")

       with gr.Column():
         img2text_output = gr.Textbox(
                                 label="Generated text prompt", 
                                 lines=4,
                                 elem_id="translated"
                             )
         with gr.Row():
             diffuse_btn = gr.Button(value="Diffuse it!")
       with gr.Column(elem_id="generated-gallery"):
         sd_output = gr.Gallery().style(grid=2, height="auto")
         with gr.Group(elem_id="share-btn-container"):
             community_icon = gr.HTML(community_icon_html, visible=False)
             loading_icon = gr.HTML(loading_icon_html, visible=False)
             share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)

     see_prompts.click(get_prompts, 
                             inputs = input_img, 
                             outputs = [
                                 img2text_output
                             ])
     diffuse_btn.click(get_images, 
                           inputs = [
                               img2text_output
                               ], 
                           outputs = [sd_output, community_icon, loading_icon, share_button]
                           )
     share_button.click(None, [], [], _js=share_js)

if not is_colab:
  demo.queue(concurrency_count=4)
demo.launch(debug=is_colab, share=is_colab)
demo.queue(max_size=25).launch()
demo.queue()
demo.launch()