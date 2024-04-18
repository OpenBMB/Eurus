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
    sampling_params = SamplingParams(max_tokens=2048,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=[],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = ["```python\n" +  match_code(output.outputs[0].text) + "\n```" for output in outputs]
    return completions


from fastchat.conversation import get_conv_template
def make_conv(example, model_type):
    conv = get_conv_template(model_type).copy() # only mistral currently
    msg = example["prompt_sft"] + "\nYou need first write a step-by-step outline and then write the code."
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt() + " ```python\n"

samples = []
del problems["start_time"]
problems["instruction"] = problems.apply(lambda row: make_conv(row, args.model_type), axis=1)

completions = generate_sample_batch(problems["instruction"])
problems["output"] = completions

samples = problems.to_dict(orient="records")

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)
