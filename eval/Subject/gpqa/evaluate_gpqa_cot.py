from typing import Dict, Any
import os
import json
from tqdm import tqdm
from datetime import datetime
import openai
from time import sleep
import argparse
from util import *
from datasets import Dataset, load_dataset
import pandas as pd
from vllm import LLM, SamplingParams
import torch
import time
from collections import namedtuple

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])



parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
# meta-llama/Llama-2-7b-chat-hf
# parser.add_argument("--input_data", type=str, default="./theorem_qa.json")
parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--model_type", type=str, default='mistral')

args = parser.parse_args()


llm = LLM(
    model=args.model,
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
)

def generate_sample_batch(question_list):
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0.0,
                                    top_p=1,
                                    n=1,
                                    stop=["question:"],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text for output in outputs]
    return completions


# def create_reader_request(example: Dict[str, Any]) -> str:
#     string = f"Question: {example['Question']}"
#     return string


# Copied from: https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py#L197
def load_examples(data):
    import random
    seed = 0
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']))
        return example

    return [shuffle_choices_and_create_example(row) for _, row in data.iterrows()]


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(example, model_type, prev_output=None):
    prompt_round1 = f"What is the correct answer to this question: {example.question}"
    prompt_round1 += f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt_round1 += "\nLet's think step-by-step:"
    prompt_round2 = '\n\nBased on the above, what is the single, most likely answer choice? Answer in the format \"The correct answer is (insert answer here)\".'

    if prev_output is None:
        prompt = prompt_round1
    else:
        prompt = prompt_round1 + prev_output + prompt_round2

    msg = [{"role": "user", "content": prompt},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out
    


if __name__ == "__main__":

    
    test_set = load_dataset('Idavidrein/gpqa', 'gpqa_main')['train'].to_pandas()[:3]
    test_set['example'] = load_examples(test_set)
    test_set["prompt_round1"] = test_set['example'].apply(lambda example: make_conv(example, args.model_type))
    
    # Adapted from: https://github.com/idavidrein/gpqa/blob/main/baselines/utils.py#L48
    # First round: Think step-by-step
    test_set["completion_round1"] = generate_sample_batch(test_set["prompt_round1"].tolist())

    # Second round: Extract final answer
    test_set["prompt_round2"] = test_set.apply(lambda row: make_conv(row['example'], args.model_type, row['completion_round1']), axis=1)
    test_set["completion"] = generate_sample_batch(test_set["prompt_round2"].tolist())
    test_set.to_json(os.path.join(args.save_dir, "gpqa_completion.json"))
    
    answered_set = dict()
    correct, wrong = 0, 0

    output_filename = os.path.join(args.save_dir, "gpqa_output.json")
    writer = open(output_filename, 'w')
    accuracy = []
    index_mapping = ['(a)', '(b)', '(c)', '(d)']

    for example in test_set.to_dict(orient="records"):
        
        if answered_set and example['id'] in answered_set:
            writer.write(answered_set[example['id']] + '\n')
            continue

        result = example["completion"]
        _, prediction = match_answer(result)
        prediction = postprocess_number(prediction)
        reference = index_mapping[example["example"].correct_index]

        verifier = GPQATask(id=example["Record ID"], 
                            prompt=example["prompt_round2"], 
                            reference=reference,
                            answer_type='Option')
        acc = verifier.success(prediction)
        tmp = {
            'id': example['Record ID'],
            'question': example['prompt_round2'],
            'prediction': prediction,
            'answer': reference,
            'rationale': result,
            'answer_type': 'Option',
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

