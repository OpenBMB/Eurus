import os
import re
import time
import openai
import argparse
import traceback
import fire
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset


import sys
sys.path.append("../..")
from utils.data import write_jsonl, read_problems, HUMAN_EVAL
from utils.evaluation import evaluate_functional_correctness


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--save_dir",default="./" ,type=str)
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--model_type", type=str, default='mistral')
# for pass@1
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/docs/README.md?plain=1#L44
parser.add_argument("--temperature", type=float, default=0.2)
args = parser.parse_args()

problems = read_problems()
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/bigcode_eval/tasks/humaneval.py#L54C13-L54C87
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

def match_code(s):
    if '```python' in s:
        pattern = r'```python(.*?)```'
        return re.findall(pattern, s, re.DOTALL)[0]
    elif '```' in s:
        pattern = r'```(.*?)```'
        return re.findall(pattern, s, re.DOTALL)[0]
    else:
        return s

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=[],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text.split('```')[0] for output in outputs]
    return completions

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature

from fastchat.conversation import get_conv_template
def make_conv(example,model_type):
    conv = get_conv_template(model_type).copy() # only mistral currently
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                f"```python\n"
                f"{example['prompt']}\n"
                f"```\n"
            )
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt() + f" ```python\n{example['prompt']}"

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    results = {k:v*100 for k,v in results.items()}
    print(results)


samples = []
problems = Dataset.from_pandas(pd.DataFrame(problems).T)
problems = problems.map(lambda x: {"signature": make_signature(x)}, cache_file_name="../../cache/human_eval", load_from_cache_file=False)
problems = problems.map(lambda x: {"instruction": make_conv(x, args.model_type)}, cache_file_name="../../cache/human_eval", load_from_cache_file=False)

completions = generate_sample_batch(problems["instruction"])
problems = problems.add_column("completion", completions)
problems = problems.map(lambda x: {"completion": x["prompt"] + x["completion"]})
samples = problems.to_pandas().to_dict(orient="records")

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)

fire.Fire(entry_point(output_filepath))

