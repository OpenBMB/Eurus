import os
import re
import time
import openai
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset
import json

import sys
sys.path.append("../..")
from utils.data import write_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--save_dir", type=str,default="./")
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--model_type", type=str, default='mistral')
# for pass@1
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/docs/README.md?plain=1#L44
parser.add_argument("--temperature", type=float, default=0.)
args = parser.parse_args()

problems = pd.read_json("leetcode-test.json", lines=True)
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/bigcode_eval/tasks/humaneval.py#L54C13-L54C87
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

def match_code(s):
    pattern = r'```python(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        return sol[0]
    
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        return sol[0]
    
    return s.split('```')[0]

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    global EOS_TOKEN
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=[EOS_TOKEN],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = ["```python\n" +  match_code(output.outputs[0].text) + "\n```" for output in outputs]
    return completions

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(args.model)
# def make_conv(example):
#     prompt = example["prompt_sft"] + "\nYou need first write a step-by-step outline and then write the code."
#     msg =  [{"role": "user", "content": prompt},]
#     msg.append({"role": "assistant", "content": "```python\n" if "eurus-70b" not in args.model.lower() else ""})

#     out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
#     out = out.rstrip(tokenizer.eos_token).strip()
#     return out

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
global EOS_TOKEN
EOS_TOKEN = tokenizer.eos_token
def make_conv_hf(example, tokenizer):
    # msg = [
    #     {"role": "user", "content": example["prompt_sft"] + "\nYou need first write a step-by-step outline and then write the code."}
    # ]
    msg = [
        {"role": "user", "content": example["prompt_sft"]}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat + "```python"


samples = []
del problems["start_time"]
problems["instruction"] = problems.apply(lambda row: make_conv_hf(row, tokenizer), axis=1)

completions = generate_sample_batch(problems["instruction"])
problems["output"] = completions

#problems.to_json(os.path.join(args.save_dir, "completions.jsonl"), orient="records", indent=4)

samples = problems.to_dict(orient="records")

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)
