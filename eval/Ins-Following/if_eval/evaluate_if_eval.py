import os
import time
import json
import openai
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--input_data", type=str, default="./input_data.jsonl")
parser.add_argument("--save_path", type=str, default="./input_response_data.jsonl")
parser.add_argument("--model_type", type=str, default='mistral')

args = parser.parse_args()


from vllm import LLM, SamplingParams
import torch
def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count()
    )
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0.7,
                                    n=1,
                                    stop=["Question:"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text.strip() for output in outputs]
    return completions

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(question):
    msg =  [{"role": "user", "content": question},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    out = out.lstrip(tokenizer.bos_token)
    return out

if __name__ == "__main__":
    

    dataset = pd.read_json(args.input_data, lines=True)
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(lambda x: {"instruction": make_conv(x["prompt"])}, cache_file_name="../../cache/if_eval", load_from_cache_file=False)
    completions = generate_sample_batch(dataset["instruction"])
    dataset = dataset.add_column("response", completions)

    dataset = dataset.to_json(args.save_path)
