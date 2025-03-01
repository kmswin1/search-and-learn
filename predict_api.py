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
# subquestions=[]
# captions=[]
# questions=[]
# with open("results_exp/"+file+"_"+dataset+"-step2.jsonl", "r") as f:
#     for line in f:
#         line = json.loads(line)
#         subquestions.append([line["question"]] + line["subquestions"])
#         captions.append(line["captions"])

# with open("results_exp/"+file+"_"+dataset+"-step3.jsonl", "r") as f:
#     for line in f:
#         line = json.loads(line)
#         questions.append(line["question"])
        
with open("results/"+file+"_"+dataset+".jsonl", "w") as wf:
    for i, meta in enumerate(tqdm(ds)):
        try:
            question = meta["query"] if dataset == "mathvista" else meta["question"]
            # if question in questions:
            #     continue
            answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
            image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
            image = pil_to_base64(image)
            ## Step 1
            system_prompt = "Your task is to answer the question about the image. When you’re ready to answer, conclude using the format \"Final answer: \""
            prompt = system_prompt + "\nQuestion: "+ question
            pred = call(prompt, image)
            
            res = {}
            res["question"] = question
            res["answer"] = answer
            # res["captions"] = captions[i]
            # res["scene_graphs"] = scene_graphs
            res["prediction"] = pred if type(pred) == str else pred[0]
            wf.write(json.dumps(res)+"\n")

        except Exception as e:
            print (e)
            res = {}
            res["question"] = question
            res["answer"] = answer
            # res["captions"] = []
            # res["scene_graphs"] = []
            res["prediction"] = []
            wf.write(json.dumps(res)+"\n")
            pass


# with open("results_exp/"+file+"_"+dataset+"-step1.jsonl", "w") as wf:
#     for i, meta in enumerate(ds):
#         try:
#             image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
#             question = meta["query"] if dataset == "mathvista" else meta["question"]
#             answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
#             conversation = [
#                             {"role": "user", "content": [
#                                 {"type": "image"},
#                                 {"type": "text", "text": question + "\nLet's break down the question into easier sub-questions. Write subquestions as 1.[subquestion]\n2.[subquestion]..."} 
#                             ]}
#                         ]


#             # Preprocess the inputs
#             text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#             # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

#             inputs = processor(
#                 text=[text_prompt], images=[image], padding=True, return_tensors="pt"
#             )
#             inputs = inputs.to("cuda")
#             # Inference: Generation of the output
#             # output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
#             output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, do_sample=True, top_p=0.95)
#             generated_ids = [
#                 output_ids[len(input_ids) :]
#                 for input_ids, output_ids in zip(inputs.input_ids, output_ids)
#             ]
#             output_text = processor.batch_decode(
#                 generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#             )

#             pred = output_text
#             res = {}
#             res["question"] = question
#             res["answer"] = answer
#             res["subquestions"] = pred if type(pred) == str else pred[0]
#             # res["prediction"] = pred if type(pred) == str else pred[0]
#             wf.write(json.dumps(res)+"\n")

#             del inputs
#             del output_ids
#         except Exception as e:
#             print (e)
#             res = {}
#             res["question"] = question
#             res["answer"] = answer
#             res["subquestions"] = ""
#             wf.write(json.dumps(res)+"\n")
#             pass
