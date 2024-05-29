import json
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import os
import torch
import openai
import argparse
from vllm import LLM, SamplingParams
import time
import re

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("./scripts/eval")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--input_data", type=str, default="./new_mbpp.json")
parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--model_type", type=str, default='mistral')

args = parser.parse_args()

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

STOP_WORDS =["\nassert", "assert", "\ndef "]

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
    sampling_params = SamplingParams(max_tokens=512,
                                    temperature=0.0,
                                    n=1,
                                    stop=STOP_WORDS)

    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = []
    for completion in [output.outputs[0].text.split('```')[0] for output in outputs]:
        completion = completion.split('```')[0]
        completions.append(completion)
    return completions

def make_signature(code):
    signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
    signature = signature.lstrip("def ").replace(" ", "").rstrip(":").strip().replace(",", ", ")
    assert ":" not in signature
    return signature

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(signature, description, test_list):
    description = description.split(" https://www.")[0]
    #testcase = "\n>>> ".join(test_list)
    testcase = test_list[0]
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                #f"```python\n"
                f"{description}\n"
                f">>> {testcase}\n"
                #f"```\n"
            )
    msg =  [{"role": "user", "content": prompt}]
    # if "eurus-70b" not in args.model.lower():
    #msg.append({"role": "assistant", "content": " ```python\ndef"})
    #     out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    # else:
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # out = out.lstrip(tokenizer.bos_token).strip()
    # out = out.rstrip(tokenizer.eos_token).strip()
    return out+"```python\ndef"

import contextlib
import signal
class TimeoutException(Exception):
    pass
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def exec_helper(code):
    with time_limit(3):
        exec(compile(code, filename="mbpp", mode='exec'), globals())

def evaluate(dataset):
    correct = 0
    format_error = 0
    exec_error = 0

    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # remove texts
        code = completion.split("\n")
        code_ = []
        for c in code:
            if len(c.lstrip()) == len(c) and not c.startswith("def"):
                continue
            code_.append(c)
        code = "\n".join(code_)

        function = code
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([
            function,
            test_cases,
        ])

        # define function
        try:
            exec_helper(function)
        except Exception as e:
            format_error += 1
            continue           

        try:
            # run test case
            exec_helper(test_cases)
            exec_helper(test_run)
        except:
            exec_error += 1
            continue
        else:
            correct += 1
    return 100 * correct / len(dataset), 100 * exec_error / len(dataset), 100 * format_error / len(dataset)


if __name__ == "__main__":
    

    dataset = pd.read_json(args.input_data, lines=False)
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    for signature in dataset["signature"]:
        STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv(row["signature"], row["prompt"], row["test_list"]), axis=1)
    completions = generate_sample_batch(dataset["prompt"].tolist())
    dataset["completion"] = completions
    del dataset["source_file"]
    dataset["completion"] = dataset.apply(lambda row: "def" + row["completion"] if "def" not in row["completion"] else row["completion"], axis=1)
    dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"))


    accuracy, exec_error, format_error = evaluate(dataset)
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error})
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error}, file=f)
