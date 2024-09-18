import os
import re
import argparse
import pandas as pd
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

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count()
    )
    global EOS_TOKEN
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=[EOS_TOKEN],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    #completions = [match_code(output.outputs[0].text) for output in outputs]
    completions = [output.outputs[0].text.split('```')[0] for output in outputs]
    return completions

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
global EOS_TOKEN
EOS_TOKEN = tokenizer.eos_token
def make_conv(example):
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
                # f"{description}\n"
                f"{example['prompt']}"
                f"```\n"
            )

    msg =  [{"role": "user", "content": prompt}]
    # if "eurus-70b" not in args.model.lower():
    #msg.append({"role": "assistant", "content": "```Python\n"})
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # else:
    #      out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    #out = out.rstrip(tokenizer.eos_token).strip()
    return out + " ```python\ndef"
    # return out



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
    results = evaluate_functional_correctness(sample_file, k=k, n_workers=n_workers, timeout=timeout, problem_file=problem_file)
    results = {k:v*100 for k,v in results.items()}
    print(results)


samples = []
problems = Dataset.from_pandas(pd.DataFrame(problems).T)
problems = problems.map(lambda x: {"signature": make_signature(x)}, cache_file_name="../../cache/human_eval", load_from_cache_file=False)
problems = problems.map(lambda x: {"instruction": make_conv(x)}, cache_file_name="../../cache/human_eval", load_from_cache_file=False)

completions = generate_sample_batch(problems["instruction"])
problems = problems.add_column("completion", completions)
problems = problems.map(lambda x: {"completion": "def "+ x["completion"]})
# problems = problems.map(lambda x: {"completion": x["prompt"] + x["completion"]})
samples = problems.to_pandas().to_dict(orient="records")

os.makedirs(args.save_dir, exist_ok=True)
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)

entry_point(output_filepath)

