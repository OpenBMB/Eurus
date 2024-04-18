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
from utils.math_equivalence import is_equiv
from utils.util import clean_numbers, last_boxed_only, last_boxed_only_string
from utils.grader import math_equal
from utils.python_interpreter import postprocess_completions




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


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

            i += 1

        if left_brace_idx is None or right_brace_idx is None:
            return None

        return string[left_brace_idx + 1: right_brace_idx].strip()

def match_answer(response):
    is_matched = False
    ans_marker = 'answer:\n'
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    ans_marker = 'answer: '
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Grade
    return response


from fastchat.conversation import get_conv_template
def make_conv(question, model_type):
    conv = get_conv_template(model_type).copy() # only mistral currently
    msg = "Tool available:\n[1] Python intepreter\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment.\n"
    msg += "Solve the following math problem step-by-step.\nSimplify your answer as much as possible.\n"
    msg += question
    # add question
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()



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
                model_output = match_answer(model_output)
                outputs.append(model_output)
                answers.append(answer)

                try:
                    # equiv = is_equiv(model_output, answer)
                    equiv = math_equal(model_output, answer)
                except:
                    equiv = False
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
                model_output = match_answer(model_output)
                outputs.append(model_output)
                try:
                    # equiv = is_equiv(model_output, answer)
                    equiv = math_equal(model_output, answer)
                except:
                    equiv = False
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        elif file == "gsmplus_test.json":
            all_problems = pd.read_json(os.path.join(args.data_dir, file))
            conv_list = [ make_conv(all_problems[problem]["perturbation_questions"][sub_problem]["question"],args.model_type) for problem in all_problems for sub_problem in all_problems[problem]["perturbation_questions"]  ]
            answers = [all_problems[problem]["perturbation_questions"][sub_problem]["answer"] for problem in all_problems for sub_problem in all_problems[problem]["perturbation_questions"] ]
            completions = generate_sample_batch(conv_list,llm)
            for answer, model_output in zip(answers, completions):
                model_output = match_answer(model_output)
                outputs.append(model_output)
                try:
                    # equiv = is_equiv(model_output, answer)
                    equiv = math_equal(model_output, answer)
                except:
                    equiv = False
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
                model_output = match_answer(model_output)
                outputs.append(model_output)
                try:
                    # equiv = is_equiv(model_output, answer)
                    equiv = math_equal(model_output, answer)
                except:
                    equiv = False
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="./data")
    parser.add_argument("--save_dir", "-s", type=str, default="ui_results")
    parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
    parser.add_argument("--model_type", type=str, default='mistral')
    args = parser.parse_args()
    run(args)
