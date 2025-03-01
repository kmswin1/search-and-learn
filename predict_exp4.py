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
model_name=sys.argv[1]
dataset=sys.argv[2]


if dataset == "mathvista":
    ds = load_dataset("AI4Math/MathVista")["testmini"]
elif dataset == "mathvision":
    ds = load_dataset("MathLLMs/MathVision")["test"]
elif dataset == "clevrmath":
    ds = load_dataset("dali-does/clevr-math", "general", split="test")
elif dataset == "mathverse":
    ds = load_dataset("AI4Math/MathVerse", "testmini")["testmini"]

from transformers import (MllamaForConditionalGeneration, 
                        Qwen2VLForConditionalGeneration, 
                        AutoProcessor,
                        LlavaForConditionalGeneration,
                        LlavaNextForConditionalGeneration,)

# default: Load the model on the available device(s)
if "qwen" in model_name.lower():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        device_map="auto"
        )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
elif "llama" in model_name.lower():
    model = MllamaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
elif "llava-1.5" in model_name.lower():
    model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
elif "llava-v1.6" in model_name.lower():
    model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side='left'

file = model_name.split("/")[-1]
subquestions=[]
captions=[]
questions=[]
scene_graphs=[]
filtered=[]
with open("results_exp/"+file+"_"+dataset+"-step3.jsonl", "r") as f:
    for line in f:
        line = json.loads(line)
        subquestions.append([line["question"]] + line["subquestions"])
        captions.append(line["captions"])
        scene_graphs.append(line["scene_graphs"])

        
with open("results_exp/"+file+"_"+dataset+"-step4.jsonl", "w") as wf:
    for i, meta in enumerate(tqdm(ds)):
        try:
            question = meta["query"] if dataset == "mathvista" else meta["question"]
            # if question in questions:
            #     continue
            filtered_scene_graphs = []
            text_prompts = []
            images = []
            answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
            image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
            for s, c in zip(scene_graphs[i], captions[i]):
                prompt = "Does this scene graph have a valid json format and good at representing the above image and its descriptions? answer yes or no."
                conversation = [
                            {"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"Image Description: {c}\nScene graph: {s}\n{prompt}"} 
                            ]}
                        ]


                # Preprocess the inputs
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                # # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
                text_prompts.append(text_prompt)
                images.append(image)

            inputs = processor(
                text=text_prompts, images=images, padding=True, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            # output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
            output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True, top_p=0.9)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            for j, pred in enumerate(output_text):
                if "yes" in pred.lower():
                    filtered_scene_graphs.append(scene_graphs[i][j])
            
            del inputs
            del output_ids

            res = {}
            res["question"] = question
            res["answer"] = answer
            res["subquestions"] = subquestions[i]
            res["captions"] = captions[i]
            res["scene_graphs"] = scene_graphs[i]
            res["filtered_scene_graphs"] = filtered_scene_graphs
            res["prediction"] = pred if type(pred) == str else pred[0]
            wf.write(json.dumps(res)+"\n")

        except Exception as e:
            print (e)
            res = {}
            res["question"] = question
            res["answer"] = answer
            res["subquestions"] = []
            res["captions"] = []
            res["scene_graphs"] = []
            res["filtered_scene_graphs"] = []
            res["prediction"] = ""
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
