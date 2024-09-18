import os
import time
import traceback
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


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", "-d", type=str,
                    default="./data")
parser.add_argument("--save_dir", "-s", type=str, default="result")
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--model_type", type=str, default='mistral')
args = parser.parse_args()


import sys
sys.path.append("../..")
from utils import evaluate_math

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

other_set = ["SVAMP.json","gsmplus_test.json"]


def generate_sample_batch(question_list,llm):

    sampling_params = SamplingParams(max_tokens=1024,
                                    temperature=0,
                                    stop=["\n###\nProblem: "],)
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text for output in outputs]
    return completions


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


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(question, model_type):
    prompt = "Solve the following math problem step-by-step.\n" + "Simplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n" + question
    # add question
    msg =  [{"role": "user", "content": prompt},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return out
    


def run_math_chat(args, max=-1):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        #tensor_parallel_size=2,
    )
    total_result = []
    deepmind_result = [0,0]
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
            for problem_data, model_output in zip(test_problem, completions):
                answer = problem_data["Output Answer"][0]
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                answers.append(answer)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        elif file == "SVAMP.json":
            all_problems = pd.read_json(os.path.join(args.data_dir, file))["Question"]
            all_body = pd.read_json(os.path.join(args.data_dir,file))["Body"]
            all_answers = pd.read_json(os.path.join(args.data_dir, file))["Answer"]
            conv_list = [ make_conv(body+". "+problem,args.model_type) for body,problem in zip(all_body,all_problems) ]
            completions = generate_sample_batch(conv_list,llm)
            answers = all_answers
            for answer, model_output in zip(all_answers, completions):
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1
        elif file == "gsmplus_test.json":
            all_problems = pd.read_json(os.path.join(args.data_dir, file))
            conv_list = [ make_conv(all_problems[problem]["perturbation_questions"][sub_problem]["question"], args.model_type) for problem in all_problems for sub_problem in all_problems[problem]["perturbation_questions"]  ]
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
            for answer, model_output in zip(all_answers, completions):
                is_matched, equiv, model_output = evaluate_math(model_output, answer)
                outputs.append(model_output)
                fnames_list.append(equiv)
                if equiv:
                    correct += 1


        output_file = os.path.join(args.save_dir, file_name+"_"+"results.txt")
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

            total = len(conv_list)
            print(file_name+" "+"Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))

            total_result.append([file_name,correct,total])
            f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))
            output_dict["overall_accuracy"] = {
                "num_correct": correct,
                "num_total": total,
                "accuracy": correct / total
            }


            class JSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.int64):
                        return int(obj)
                    return super(JSONEncoder, self).default(obj)


            with open(os.path.join(args.save_dir, file_name+"_"+"results.json"), "w") as jf:
                json.dump(output_dict, jf, cls=JSONEncoder)

    total_output_file = os.path.join(args.save_dir, "total_results.txt")
    with open(total_output_file, "w+") as f:
        for set_name,set_correct,set_total in total_result:
            f.write(set_name+" Accuracy = {}/{} = {:.3f}\n".format(set_correct, set_total, set_correct / set_total))



if __name__ == "__main__":
    


    run_math_chat(args)




