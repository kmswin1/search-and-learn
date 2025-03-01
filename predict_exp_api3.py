from datasets import load_dataset
import PIL
import sys
import json
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

model_name=sys.argv[1]
dataset=sys.argv[2]


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if dataset == "mathvista":
    ds = load_dataset("AI4Math/MathVista")["testmini"]
elif dataset == "mathvision":
    ds = load_dataset("MathLLMs/MathVision")["test"]
elif dataset == "clevrmath":
    ds = load_dataset("dali-does/clevr-math", "general", split="test")
elif dataset == "mathverse":
    ds = load_dataset("AI4Math/MathVerse", "testmini")["testmini"]

from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call(query, image):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

file = model_name.split("/")[-1]
subquestions=[]
captions=[]
with open("results_exp/"+file+"_"+dataset+"-step2.jsonl", "r") as f:
    for line in f:
        line = json.loads(line)
        subquestions.append([line["question"]] + line["subquestions"])
        captions.append(line["captions"])
        
with open("results_exp/"+file+"_"+dataset+"-step3.jsonl", "w") as wf:
    for i, meta in enumerate(tqdm(ds)):
        try:
            scene_graphs = []
            question = meta["query"] if dataset == "mathvista" else meta["question"]
            # if question in questions:
            #     continue
            answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
            image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
            image = pil_to_base64(image)
            for q, c in zip(subquestions[i], captions[i]):
                # ## Step 3
                prompt = """
                    For the provided image and its associated image description, generate a scene graph in JSON format that includes the following:
                    1. Objects that are relevant to answering the question.
                    2. Object attributes that are relevant to answering the question.
                    3. Object relationships that are relevant to answering the question.
                    """
                query = f"Image Description: {c}\n{prompt}"
                scene_graphs.append(call(query, image))

            res = {}
            res["question"] = question
            res["answer"] = answer
            res["subquestions"] = subquestions[i]
            res["captions"] = captions[i]
            res["scene_graphs"] = scene_graphs
            wf.write(json.dumps(res)+"\n")

        except Exception as e:
            print (e)
            res = {}
            res["question"] = question
            res["answer"] = answer
            res["subquestions"] = []
            res["captions"] = []
            res["scene_graphs"] = []
            wf.write(json.dumps(res)+"\n")