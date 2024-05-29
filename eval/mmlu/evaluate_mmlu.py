# Modified from https://github.com/hendrycks/test/blob/master/evaluate.py
import openai
import argparse
import traceback
import os
import json
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
# from crop import crop
from vllm import LLM, SamplingParams
import torch

openai.api_key = "INSERTYOURKEYHERE"
choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def generate_sample_batch(question_list):
    global loaded, llm
    if not loaded:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        loaded = True
    sampling_params = SamplingParams(max_tokens=1,
                                    logprobs=100,
                                    temperature=0,
                                    )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=False)
    completions = [output.outputs[0].text.strip() for output in outputs]
    return completions



def eval(args, subject, model_name, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    prompt_list = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompt_list.append(prompt)

    completions = generate_sample_batch(prompt_list)

    for i, pred in enumerate(completions):
        label = test_df.iloc[i, test_df.shape[1]-1]
        # pred = {0: "A", 1: "B", 2: "C", 3: "D"}[c]
    #     # \u2581 is the unicode character for a space in tokenizer
    #     d = c["choices"][0]["logprobs"]["top_logprobs"][-1]
    #     new_d = {
    #         k.replace("\u2581", " "): v
    #         for k, v in d.items()
    #     }
    #     c["choices"][0]["logprobs"]["top_logprobs"][-1] = new_d

    #     lprobs = []
    #     for ans in answers:
    #         try:
    #             lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
    #         except:
    #             print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
    #             lprobs.append(-100)
    #     pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
    #     probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        # all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    # all_probs = np.array(all_probs)
    # print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    model_name = args.model
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir)):
        os.mkdir(os.path.join(args.save_dir))

    # print(subjects)
    # print(args)
    print("Evaluating {} on {} subjects".format(model_name, len(subjects)))
    all_cors = []

    metrics = {
        "subject_acc": {},
        "overall_acc": None
    }
    for subject in tqdm(subjects, total=len(subjects), desc="Subjects"):
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, model_name, dev_df, test_df)
        metrics["subject_acc"][subject] = acc
        all_cors.append(cors)

        test_df["correct"] = cors
        # for j in range(probs.shape[1]):
        #     choice = choices[j]
        #     test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    metrics["overall_acc"] = weighted_acc * 100
    print("Average accuracy: {:.3f}".format(weighted_acc * 100))
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/mnt/data/user/tc_agi/ylf/eval_data/mmlu")
    parser.add_argument("--save_dir", "-s", type=str, default="/data/results/mmlu")
    parser.add_argument("--model", type=str, default="/mnt/data/user/tc_agi/user/lwd/mistral-7b/mistral-7b")
    args = parser.parse_args()
    loaded = False
    llm = None
    main(args)

