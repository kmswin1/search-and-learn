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

file = model_name.split("/")[-1]
subquestions=[]
captions=[]
with open("results_exp/"+file+"_"+dataset+"-step2.jsonl", "r") as f:
    for line in f:
        line = json.loads(line)
        subquestions.append([line["question"]] + line["subquestions"].split("\n"))
        captions.append(line["captions"])

with open("results_exp/"+file+"_"+dataset+"-step3.jsonl", "w") as wf:
    for i, meta in enumerate(tqdm(ds)):
        try:
            scene_graphs = []
            text_prompts = []
            images = []
            for c, q in zip(captions[i], subquestions[i]):
                image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
                question = meta["query"] if dataset == "mathvista" else meta["question"]
                answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
                instruction = """
                            For the provided image and its associated image description and question, generate a scene graph in JSON format that includes the following:
                            1. Objects that are relevant to answering the question.
                            2. Object attributes that are relevant to answering the question.
                            3. Object relationships that are relevant to answering the question.
                            """
                conversation = [
                                {"role": "user", "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": "Question: "+ q + "\n" + "Image caption: "+ c + "\n" + instruction} 
                                ]}
                            ]


                # Preprocess the inputs
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
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
            for pred in output_text:
                scene_graphs.append(pred)
            
            del inputs
            del output_ids

            res = {}
            res["question"] = question
            res["answer"] = answer
            res["subquestions"] = subquestions[i]
            res["captions"] = captions[i]
            res["scene_graphs"] = scene_graphs
            # res["prediction"] = pred if type(pred) == str else pred[0]
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
            pass
            
        # try:
        #     captions = []
        #     for q in subquestions[i]:
        #     image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
        #     question = meta["query"] if dataset == "mathvista" else meta["question"]
        #     answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
        #     conversation = [
        #                     {"role": "user", "content": [
        #                         {"type": "image"},
        #                         {"type": "text", "text": "Question: "+ q + "\nDescribe and extract the key information from the image to answer the question."} 
        #                     ]}
        #                 ]


        #     # Preprocess the inputs
        #     text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        #     # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

        #     inputs = processor(
        #         text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        #     )
        #     inputs = inputs.to("cuda")
        #     # Inference: Generation of the output
        #     # output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
        #     output_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True, top_p=0.9)
        #     generated_ids = [
        #         output_ids[len(input_ids) :]
        #         for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        #     ]
        #     output_text = processor.batch_decode(
        #         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #     )
        #     pred = output_text if type(output_text) == str else output_text[0]
        #     captions.append(pred)
            
        #     del inputs
        #     del output_ids

        #     res = {}
        #     res["question"] = question
        #     res["answer"] = answer
        #     res["subquestions"] = subquestions[i]
        #     res["captions"] = captions
        #     # res["prediction"] = pred if type(pred) == str else pred[0]
        #     wf.write(json.dumps(res)+"\n")

        # except Exception as e:
        #     print (e)
        #     res = {}
        #     res["question"] = question
        #     res["answer"] = answer
        #     res["subquestions"] = []
        #     res["captions"] = []
        #     wf.write(json.dumps(res)+"\n")
        #     pass

# with open("results_exp/"+file+"_"+dataset+"-step2.jsonl", "w") as wf:
#     for i, meta in enumerate(ds):
#         try:
#             image = meta["decoded_image"] if dataset != "clevrmath" and dataset != "mathverse" and dataset != "seed" else meta["image"]
#             question = meta["query"] if dataset == "mathvista" else meta["question"]
#             answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
#             # system_prompt = "Your task is to answer the question about the image. When you’re ready to answer, conclude using the format \"Final answer: \""
#             # system_prompt = "Your task is to answer the question about the image. First, you describe about the image using format ## Image Description: [Image description]\n\n and then give step by step reasoning based on image description.\nUse this step-by-step format:\n\n\##Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nWhen you’re ready to answer, conclude using the format \"Final answer: \""
#             # system_prompt = "Solve the following image related problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: . I hope it is correct."
#             # if "qwen" in model_name.lower():
#             #     conversation = [
#             #         {
#             #             "role": "user",
#             #             "content": [
#             #                 {
#             #                     "type": "image",
#             #                 },
#             #                 {"type": "text", 
#             #                 "text":  system_prompt + "\n" + question
#             #                 },
#             #             ],
#             #         }
#             #     ]
#             # elif "llama" in model_name.lower():
#             # conversation = [
#             #                 {"role": "user", "content": [
#             #                     {"type": "image"},
#             #                     {"type": "text", "text": question+"\nLet's break down the question into easier sub-questions. Write subquestions as 1.[subquestion]\n2.[subquestion]..."}
#             #                 ]}
#             #             ]
#             # instruction="""
#             # Let's break down the question into easier sub-questions. Write subquestions as 1.[subquestion]\n2.[subquestion]...
#             # For example,
#             # Question: \"original question\"
#             # Sub-questions: 
#             # 1. \"sub-question\"
#             # 2. \"sub-question\"
#             # 3. \"sub-question\"
#             # ...
#             # """
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
