from datasets import load_dataset
import PIL
import sys
import json
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
model_name=sys.argv[1]
dataset=sys.argv[2]


if dataset == "mathvista":
    ds = load_dataset("AI4Math/MathVista")["testmini"]
elif dataset == "mathvision":
    ds = load_dataset("MathLLMs/MathVision")["test"]
elif dataset == "clevrmath":
    ds = load_dataset("dali-does/clevr-math", "general", split="test")

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
    )
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

file = model_name.split("/")[-1]
with open("results/"+file+"_"+dataset+".jsonl", "w") as wf:
    for meta in ds:
        try:
            image = meta["decoded_image"] if dataset != "clevrmath" else meta["image"]
            question = meta["query"] if dataset == "mathvista" else meta["question"]
            answer = meta["answer"] if dataset != "clevrmath" else meta["label"]
            system_prompt = "Your task is to answer the question related to image. First, you describe about the image and then give step by step reasoning based on image description.\nUse this step-by-step format:\n\n## Image Description: [Image description]\n\n##Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nWhen you’re ready to answer, conclude using the format \"Final answer: \""
            # system_prompt = "Solve the following image related problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: . I hope it is correct."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", 
                        "text":  system_prompt + question + "\nLet's think step by step."
                        },
                    ],
                }
            ]


            # Preprocess the inputs
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            output_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False)
            # output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.8, do_sample=True, top_p=0.95)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            pred = output_text[0]
            res = {}
            res["question"] = question
            res["answer"] = answer
            res["prediction"] = pred
            wf.write(json.dumps(res)+"\n")

            del inputs
            del output_ids
        except:
            pass