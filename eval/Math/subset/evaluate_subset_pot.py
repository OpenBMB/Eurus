# Adapt from https://github.com/hendrycks/math/blob/main/modeling/evaluate_gpt3.py

import os
import time
import traceback
import openai
import argparse
import numpy as np
import operator
import json
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
import re

import sys
sys.path.append("../..")
from utils import evaluate_math
from utils.python_interpreter import postprocess_completions

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", "-d", type=str, default="./data")
parser.add_argument("--save_dir", "-s", type=str, default="ui_results")
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--model_type", type=str, default='mistral')
args = parser.parse_args()


other_set = ["SVAMP.json","gsmplus_test.json"]

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def generate_sample_batch(question_list,llm):

    sampling_params = SamplingParams(max_tokens=2048,
                                    n=1,
                                    temperature=0,
                                    stop=["\n###\nProblem: "],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text.strip() for output in outputs]
    completions = postprocess_completions(completions)
    return completions



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(question, model_type):
    # prompt = "Tool available:\n[1] Python interpreter\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment.\n"
    # prompt += "Solve the following math problem step-by-step.\nSimplify your answer as much as possible.\n"
    # prompt += question
    # # add question
    # msg =  [{"role": "user", "content": prompt},]
    # out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # return out
    system = "Tool available:\n[1] Python interpreter\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. Present code in ```python```\n"
    content = "Solve the following math problem step-by-step.\n" + "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n" + question
    # add question
    
    try:
        msg = [
            {"role": "system", "content": system},
            {"role": "user", "content": content}
        ]
        chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    except:
        msg = [
            {"role": "user", "content": system+content}
        ]
        chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat



def run(args, max=-1):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    total_result = []
    files = os.listdir(args.data_dir)
    for file in files:
        if not os.path.isfile(os.path.join(args.data_dir, file)):
            continue
        file_name = str.split(file,'.')[0]
        outputs = []
        answers = []
        fnames_list = []
        correct = 0
        total = 0
        conv_list = []
        test_problem = []


        if file not in other_set:
            all_problems = pd.read_json(os.path.join(args.data_dir, file), orient='index')[0][2]
            for problem_data in all_problems:
                if problem_data["split"]=="test":
                    test_problem.append(problem_data)
                    conv_list.append(make_conv(problem_data["Input"],args.model_type))
            completions = generate_sample_batch(conv_list,llm)
            for problem_data, model_output in tqdm(zip(test_problem, completions), total=len(test_problem), desc="Matching"):
                answer = problem_data["Output Answer"][0]
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                answers.append(answer)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        elif file == "SVAMP.json":
            all_problems = pd.read_json(os.path.join(args.data_dir, file))["Question"]
            all_answers = pd.read_json(os.path.join(args.data_dir, file))["Answer"]
            all_body = pd.read_json(os.path.join(args.data_dir,file))["Body"]
            conv_list = [ make_conv(body+". "+problem,args.model_type) for body,problem in zip(all_body,all_problems) ]
            completions = generate_sample_batch(conv_list,llm)
            answers = all_answers
            for answer, model_output in tqdm(zip(all_answers, completions), total=len(all_answers), desc="Matching"):
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        elif file == "gsmplus_test.json":
            all_problems = pd.read_json(os.path.join(args.data_dir, file))
            conv_list = [ make_conv(all_problems[problem]["perturbation_questions"][sub_problem]["question"],args.model_type) for problem in all_problems for sub_problem in all_problems[problem]["perturbation_questions"]  ]
            answers = [all_problems[problem]["perturbation_questions"][sub_problem]["answer"] for problem in all_problems for sub_problem in all_problems[problem]["perturbation_questions"] ]
            completions = generate_sample_batch(conv_list,llm)
            for answer, model_output in zip(answers, completions):
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        else:
            all_problems = pd.read_json(os.path.join(args.data_dir, file))["question"]
            all_answers = pd.read_json(os.path.join(args.data_dir, file))["answer"]
            conv_list = [ make_conv(problem, args.model_type) for problem in all_problems ]
            completions = generate_sample_batch(conv_list,llm)
            answers = all_answers
            for answer, model_output in tqdm(zip(all_answers, completions), total=len(all_answers), desc="Matching"):
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1


        output_file = os.path.join(args.save_dir, file_name+"_results.txt")

        output_dict = {
            "outputs": [],
            "accuracy_by_subject_and_level": defaultdict(list),
            "accuracy_by_level": [],
            "accuracy_by_subject": [],
        }
        with open(output_file, "w+") as f:
            for k, (output, answer, equiv) in enumerate(zip(outputs, answers, fnames_list)):
                f.write("{} OUTPUT: {} | ANSWER: {} | CORRECT: {}\n".format(k, output, answer, equiv))
                output_dict["outputs"].append({
                    "output": output,
                    "answer": answer,
                    "equiv": equiv
                })

            f.write("#####################\n")

            print("#####################")
            f.write("#####################\n")
            total = len(conv_list)
            print(file_name+" Overall Accuracy = {}/{} = {:.3f}%".format(correct, total, correct/total * 100))
            f.write("Overall Accuracy = {}/{} = {:.3f}%\n".format(correct, total, correct/total * 100))
            output_dict["overall_accuracy"] = {
                "num_correct": correct,
                "num_total": total,
                "accuracy": correct/total
            }
            class JSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.int64):
                        return int(obj)
                    return super(JSONEncoder, self).default(obj)
            with open(os.path.join(args.save_dir, file_name+"_results.json"), "w") as jf:
                json.dump(output_dict, jf, cls=JSONEncoder)
    total_output_file = os.path.join(args.save_dir, "total_results.txt")
    with open(total_output_file, "w+") as f:
        for set_name,set_correct,set_total in total_result:
            f.write(set_name+" Accuracy = {}/{} = {:.3f}\n".format(set_correct, set_total, set_correct / set_total))

if __name__ == "__main__":

    
    run(args)
