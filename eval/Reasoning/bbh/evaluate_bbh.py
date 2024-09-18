import os
import json
import time
import traceback
import openai
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument('--data_filepath', type=str, default="test_prompts.json")
parser.add_argument('--output_filepath', type=str, default="./res.jsonl")
parser.add_argument("--model_type", type=str, default='mistral')
parser.add_argument('--is_cot', action='store_true')
parser.add_argument('--n_processes', type=int, default=8)
args = parser.parse_args()


assert args.data_filepath.endswith('.json')

df = pd.read_json(args.data_filepath, lines=True, orient='records')
print(f"Loaded {len(df)} examples.")

from vllm import LLM, SamplingParams
import torch
def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count()
    )
    sampling_params = SamplingParams(max_tokens=2048,
                                    temperature=0.0,
                                    n=1,
                                    stop=["\nQ:"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text.strip() for output in outputs]
    return completions

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv_hf(prompt, tokenizer):
    
    general_instruction = "Follow the given examples and answer the question.\n\n"
    prompt_list = prompt.split("\n\nQ: ") # [task_instruction, (Q\nA), (Q\nA)]
    task_instruction = prompt_list.pop(0)

    assert all("A: Let's think step by step." in p for p in prompt_list), (prompt, prompt_list)

    msg = []
    prompt_list = [p.split("A: Let's think step by step.") for p in prompt_list]
    for sample in prompt_list:
        assert len(sample) == 2, (sample, len(sample))
        q = "Q: " + sample[0]
        a = "A: Let's think step by step." + sample[-1]
        msg.append({"role": "user", "content": q.strip()})
        msg.append({"role": "assistant", "content": a.strip()})
    msg[0]["content"] = general_instruction + task_instruction + "\n" + msg[0]["content"]
    # chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    chat = tokenizer.apply_chat_template(msg[:-1], tokenize=False, add_generation_prompt=True)
    # chat = chat.lstrip(tokenizer.bos_token).strip()
    if "eurus-70b" not in args.model.lower():
        chat = chat.rstrip(tokenizer.eos_token).strip()
    return chat


df["prompt"] = df.apply(lambda row: make_conv_hf(row["text"], tokenizer), axis=1)
df["generation"] = generate_sample_batch(df["prompt"])


if args.is_cot:
    def check_cot_match(generation, reference) -> bool:
        generation = generation.lstrip().split("Q:")[0].strip()
        reference = reference.strip()
        return reference in generation
    df["match"] = df.apply(lambda row: check_cot_match(row["generation"], row["reference"]), axis=1)
else:
    def check_match(generation, reference) -> bool:
        generation = generation.lstrip()
        reference = reference.lstrip()
        return generation.startswith(reference)
    df["match"] = df.apply(lambda row: check_match(row["generation"], row["reference"]), axis=1)

exact_match_by_task = df.groupby("task_name")["match"].mean()
exact_match = df["match"].mean() * 100

df.to_json(args.output_filepath + ".outputs.jsonl", lines=True, orient='records')

with open(args.output_filepath, "w") as f:
    f.write(json.dumps({
        "exact_match": exact_match,
        "exact_match_by_task": exact_match_by_task.to_dict()
    }))

print("Exact match: ", exact_match)
print("Exact match by task: ", exact_match_by_task)
