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

import sys
sys.path.append("./scripts/eval")

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
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0.0,
                                    n=1,
                                    stop=STOP_WORDS)

    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [match_code(output.outputs[0].text) for output in outputs]
    return completions


from fastchat.conversation import get_conv_template
def make_signature(code):
    signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
    signature = signature.lstrip("def ").replace(" ", "").rstrip(":").strip().replace(",", ", ")
    assert ":" not in signature
    return signature

def make_conv(signature, description, test_list, model_type):
    conv = get_conv_template(model_type).copy() # only mistral currently
    description = description.split(" https://www.")[0]
    testcase = test_list[0]
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                f"{description}\n"
                f">>> {testcase}\n"
            )
    conv.append_message(conv.roles[0], prompt)

    conv.append_message(conv.roles[1], None)
    # sometimes you should uncomment the latter part
    return conv.get_prompt()# + f" ```python\ndef"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
    parser.add_argument("--input_data", type=str, default="./new_mbpp.json")
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--model_type", type=str, default='mistral')

    args = parser.parse_args()

    dataset = pd.read_json(args.input_data, lines=False)
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    for signature in dataset["signature"]:
        STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv(row["signature"], row["prompt"], row["test_list"], args.model_type), axis=1)
    completions = generate_sample_batch(dataset["prompt"].tolist())
    dataset["completion"] = completions
    del dataset["source_file"]
    dataset["completion"] = dataset.apply(lambda row: "def" + row["completion"] if "def" not in row["completion"] else row["completion"], axis=1)
    dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"))


    accuracy, exec_error, format_error = evaluate(dataset)
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error})
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error}, file=f)
