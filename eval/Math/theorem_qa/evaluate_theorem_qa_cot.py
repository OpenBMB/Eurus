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
    completions = [output.outputs[0].text for output in outputs]
    return completions


def create_reader_request(example: Dict[str, Any]) -> str:
    string = f"Question: {example['Question']}"
    return string


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(question, model_type):
    prompt = "Solve the following math problem step-by-step.\n" + "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n" + question
    # add question
    msg =  [{"role": "user", "content": prompt},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out


if __name__ == "__main__":

    
    test_set = pd.read_json(args.input_data)
    test_set["prompt"] = test_set.apply(lambda row: make_conv(create_reader_request(row),args.model_type), axis=1)
    completions = generate_sample_batch(test_set["prompt"].tolist())
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
        _, prediction = match_answer(result)
        prediction = postprocess_number(prediction)

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
            'rationale': result,
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

