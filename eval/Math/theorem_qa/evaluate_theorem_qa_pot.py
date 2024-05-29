from typing import Dict, Any
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai
from time import sleep
import argparse
from util import *
from datasets import Dataset
import pandas as pd
from vllm import LLM, SamplingParams
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--input_data", type=str, default="./theorem_qa.json")
parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--model_type", type=str, default='mistral')
args = parser.parse_args()


import sys
sys.path.append("../..")
from utils.python_interpreter import postprocess_completions

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0.0,
                                    top_p=1,
                                    n=1,
                                    stop=["Question:"],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    outputs = [output.outputs[0].text.strip() for output in outputs]
    completions = postprocess_completions(outputs)
    return outputs, completions


def create_reader_request(example: Dict[str, Any]) -> str:
    string = f"Question: {example['Question']}"
    return string


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(question):
    prompt = "Tool available:\n[1] Python interpreter\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment.\n"
    prompt += """Solve the following math problem step-by-step.\nSimplify your answer as much as possible. The answer can only be one of the following forms:
1. a numerical value like 0.1, no symbol at all.
2. a list of number like [2, 3, 4].
3. True/False.
4. an option like (a), (b), (c), (d)\n
"""
    # add question
    msg =  [{"role": "user", "content": prompt + question},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out

if __name__ == "__main__":

    test_set = pd.read_json(args.input_data)
    test_set["prompt"] = test_set.apply(lambda row: make_conv(create_reader_request(row)), axis=1)
    outputs, completions = generate_sample_batch(test_set["prompt"].tolist())
    test_set["output"] = outputs
    test_set["completion"] = completions
    test_set.to_json(os.path.join(args.save_dir, "theorem_qa_completion.json"))
    
    answered_set = dict()
    correct, wrong = 0, 0

    output_filename = os.path.join(args.save_dir, "theorem_qa_output.json")
    writer = open(output_filename, 'w')
    accuracy = []

    for example in test_set.to_dict(orient="records"):
        
        if answered_set and example['id'] in answered_set:
            writer.write(answered_set[example['id']] + '\n')
            continue

        result = example["completion"]
        prediction = extract_answer(result)
        prediction = postprocess_number(prediction)

        # print(result)
        # print(prediction, ' $$$$$$$$$ ', example['Answer'])
        if "exit()" in prediction:
            acc = False
        else:
            verifier = TheoremqaTask(id=example["id"], 
                                    prompt=example["Question"], 
                                    reference=example["Answer"], 
                                    answer_type=example["Answer_type"])
            acc = verifier.success(prediction)
        tmp = {
            'id': example['id'],
            'question': example['Question'],
            'prediction': prediction,
            'answer': example['Answer'],
            'output': example["output"],
            'completion': example['completion'],
            'answer_type': example['Answer_type'],
            "is_correct": acc,
            }
        writer.write(json.dumps(tmp) + '\n')
        
        
        accuracy.append(acc)

    writer.close()
    print()

    accuracy = sum([1 if acc else 0 for acc in accuracy]) / len(accuracy)
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print({"accuracy": accuracy * 100})
        print({"accuracy": accuracy * 100}, file=f)

