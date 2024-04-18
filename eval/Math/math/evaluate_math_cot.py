# Adapt from https://github.com/hendrycks/math/blob/main/modeling/evaluate_gpt3.py

import os
import time
import traceback
import openai
import argparse
import numpy as np
import operator
import json
import tqdm
import pandas as pd
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
import re

import math

import sys
sys.path.append("../..")
from utils.math_equivalence import is_equiv
from utils.util import clean_numbers, last_boxed_only, last_boxed_only_string
from utils.grader import math_equal

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0,
                                    stop=["\n###\nProblem: "],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text for output in outputs]
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
    return is_matched, response


from fastchat.conversation import get_conv_template
def make_conv(question, model_type):
    conv = get_conv_template(model_type).copy() # only mistral currently
    msg = "Solve the following math problem step-by-step.\n" + "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n" + question
    # add question
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

    

def run(args, max=-1):
    outputs = []
    answers = []
    types = []
    levels = []
    matches = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    
    all_problems = pd.read_json(os.path.join(args.data_dir, "math_test_cleaned.json")).to_dict(orient="records")
    completions = generate_sample_batch([make_conv(problem_data["problem"], args.model_type) for problem_data in all_problems])


    for problem_data, model_output in zip(all_problems, completions):

        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        answer = problem_data["expected_answer"]

        levels.append(prob_level)
        types.append(prob_type)
        is_matched, model_output = match_answer(model_output)
        matches.append(is_matched)
        outputs.append(model_output)
        answers.append(answer)
        
        try:
            if "\pi" in model_output or "\pi" in answer:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
                equiv = any(equivs) 
            else:
                equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False
        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        if equiv:
            correct += 1
    
    output_file = os.path.join(args.save_dir, "results.txt")
    
    output_dict = {
        "outputs": [],
        "accuracy_by_subject_and_level": defaultdict(list),
        "accuracy_by_level": [],
        "accuracy_by_subject": [],
    }
    print("Match rate: ", np.mean(matches))
    with open(output_file, "w+") as f:
        for k, (output, answer, prob_type, prob_level, match, equiv) in enumerate(zip(outputs, answers, types, levels, matches, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | MATCH: {} | CORRECT: {}\n".format(k, prob_type, prob_level, output, answer, match, equiv))
            output_dict["outputs"].append({
                "type": prob_type,
                "level": prob_level,
                "output": output,
                "answer": answer,
                "match": match,
                "equiv": equiv
            })
        

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                
                output_dict["accuracy_by_subject_and_level"][subject].append({
                    "level": level,
                    "num_correct": np.sum(cors_list),
                    "num_total": len(cors_list),
                    "accuracy": np.mean(cors_list)
                })

        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_level"].append({
                "level": level,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })

        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_subject"].append({
                "subject": subject,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })
        print("#####################")
        f.write("#####################\n")
        total = len(all_problems)
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total * 100))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total * 100))
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
        with open(os.path.join(args.save_dir, "results.json"), "w") as jf:
            json.dump(output_dict, jf, cls=JSONEncoder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="./")
    parser.add_argument("--save_dir", "-s", type=str, default="./")
    parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
    parser.add_argument("--model_type", type=str, default='mistral')
    args = parser.parse_args()
    run(args)
