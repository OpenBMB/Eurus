# copy from https://github.com/xingyaoww/mint-bench/blob/main/mint/tasks/reasoning.py
import numpy as np
from inspect import isfunction
from typing import *

import re
import ast
from sympy import Rational
import traceback


def extract_code(result: str) -> str:
    lines = []
    within_func = False
    for line in result.split('\n'):
        if line.startswith('`'):
            continue

        if line.startswith('def '):
            within_func = True

        if line.startswith('import') or line.startswith('from'):
            lines.append(line)

        if within_func:
            lines.append(line)

        if line.startswith('  return') or line.startswith('    return'):
            within_func = False

    for line_no in range(len(lines) - 1, -1, -1):
        if 'return ' not in lines[line_no]:
            del lines[line_no]
        else:
            break
    result = '\n'.join(lines)
    return result

def match_answer(response):
    is_matched = False
    ans_marker = 'The correct answer is :\n'
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]
            
    ans_marker = 'The correct answer is : '
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
        
def extract_answer(result: str) -> str:
    prediction = result.strip().strip('\n').split('\n')[-1]
    tmp = ''
    for entry in prediction.split(' ')[::-1]:
        if entry == 'is' or entry == 'be' or entry == 'are' or entry.endswith(':'):
            break
        tmp = entry + ' ' + tmp
    prediction = tmp.strip().strip('.')
    return prediction


def postprocess_number(prediction):
    if isinstance(prediction, set):
        prediction = list(prediction)
    elif isinstance(prediction, np.complex128):
        prediction = prediction.real
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    elif isinstance(prediction, complex):
        prediction = prediction.real
    elif isinstance(prediction, list):
        prediction = [float(x) for x in prediction]
    elif 'sympy' in str(type(prediction)):
        prediction = float(prediction)
    elif isfunction(prediction):
        prediction = None
    
    return prediction


impossible_questions = ['jianyu_xu/integer_programming_1.json', 'tonyxia/totient6.json']



def compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        return False

    if isinstance(gt, float):
        return within_eps(pred=p, gt=gt)
    else:
        return round(p) == gt


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def parse_number_list(s: str):
    # Check if the string is a valid list by trying to parse it
    parsed_list = ast.literal_eval(s)
    return parsed_list


def is_number(string):
    pattern = r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$"
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r"^[-+]?\d+(\.\d+)?e[-]?\d+$"
    match = re.match(pattern, string)
    return bool(match)


def contain_num_and_str(string):
    pattern_str = r"[a-zA-Z]"
    pattern_num = r"[0-9]"
    return bool(re.search(pattern_str, string) and re.search(pattern_num, string))


def is_number(string):
    pattern = r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$"
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r"^[-+]?\d+(\.\d+)?e[-]?\d+$"
    match = re.match(pattern, string)
    return bool(match)

class GPQATask():
    """Subclass of Task for multiple choice tasks."""

    task_name = "reasoning"

    def __init__(self, id, prompt: str, reference: str, **kwargs):
        self._id = id
        self._prompt = (
            "Answer the following question with a number, a list of numbers or True or False. "
            + prompt.strip()
        )
        self._reference = reference
        self._answer_type = kwargs.get("answer_type")

    def extract_answer(self, solution: str) -> Optional[Any]:
        """Extract the answer from the given solution."""
        prediction = solution.lower()
        # Following the preprocessing steps from TheoremQA
        # https://github.com/wenhuchen/TheoremQA/blob/123e36beaaa97c01f28a582f13c4f77a6822c199/predict_accuracy.py#L170

        # Preprocessing the string [Stage 1]
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else ""

        # Replace special tokens
        if "=" in prediction:
            prediction = prediction.split("=")[-1].strip()
        if "≈" in prediction:
            prediction = prediction.split("≈")[-1].strip()
        if "`" in prediction:
            prediction = prediction.replace("`", "")
        if "$" in prediction:
            prediction = prediction.replace("$", "")
        if "°" in prediction:
            prediction = prediction.replace("°", "")

        # Detect the boolean keyword in the generation
        if prediction.lower() in ["true", "yes", "false", "no"]:
            if prediction.lower() == "true" or prediction.lower() == "yes":
                prediction = "True"
            else:
                prediction = "False"
        if "True" in prediction or "False" in prediction:
            prediction = "True" if "True" in prediction else "False"

        # Detect the approximation keyword
        if "approximately" in prediction:
            prediction = prediction.replace("approximately", "").strip()
        if " or " in prediction:
            prediction = prediction.split(" or ")[0]

        # Drop the units before and after the number
        if re.match(r"[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$", prediction):
            prediction = re.search(
                r"([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$", prediction
            ).group(1)
        if re.match(r"[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$", prediction):
            prediction = re.search(
                r"[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$", prediction
            ).group(1)
        if re.match(r"[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$", prediction):
            prediction = re.search(
                r"([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$", prediction
            ).group(1)
        if re.match(r"[^-+\d]{1,2}(?:[\d,]*\.*\d+)$", prediction):
            prediction = re.search(
                r"[^-+\d]{1,2}((?:[\d,]*\.*\d+))$", prediction
            ).group(1)

        # Preprocessing the number [Stage 1]
        if "10^" in prediction:
            prediction = re.sub(r"10\^(-?\d+)", r"math.pow(10, \1)", prediction)
        if " x " in prediction:
            prediction = prediction.replace(" x ", "*")
        if " × " in prediction:
            prediction = prediction.replace(" × ", "*")
        if is_number(prediction):
            prediction = prediction.replace(",", "")

        # Preprocessing the option [Stage 3]
        if "a)" in prediction or "a )" in prediction or prediction.lower().strip() == "a":
            prediction = "(a)"
        if "b)" in prediction or "b )" in prediction or prediction.lower().strip() == "b":
            prediction = "(b)"
        if "c)" in prediction or "c )" in prediction or prediction.lower().strip() == "c":
            prediction = "(c)"
        if "d)" in prediction or "d )" in prediction or prediction.lower().strip() == "d":
            prediction = "(d)"

        if (
            "(a)" in prediction
            or "(b)" in prediction
            or "(c)" in prediction
            or "(d)" in prediction
        ):
            prediction = '"' + re.search(r"\([a-d]\)", prediction).group(0) + '"'

        # # If the prediction is empty, use dummy '0'
        # if not prediction:
        #     prediction = "0"

        # # Converting the string answer to a number/list/bool/option
        # try:
        #     prediction = eval(prediction)
        # except Exception as e:
        #     '''
        #     print(
        #         f"[TASK] Failed to convert the answer: {prediction}\n{traceback.format_exc()}"
        #     )
        #     '''
        #     return None  # failed to convert the answer
        # If the prediction is empty, use dummy '0'
        if prediction:
            # Converting the string answer to a number/list/bool/option
            try:
                prediction = eval(prediction)
            except Exception as e:
                '''
                print(
                    f"[TASK] Failed to convert the answer: {prediction}\n{traceback.format_exc()}"
                )
                '''
                return None  # failed to convert the answer

        # Performing common type conversion
        if isinstance(prediction, (set, tuple)):
            prediction = list(prediction)
            if isinstance(prediction[0], complex):
                prediction = [tmp.real for tmp in prediction]
            elif isinstance(prediction[0], Rational):
                prediction = [float(tmp) for tmp in prediction]
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        else:
            if isinstance(prediction, complex):
                prediction = prediction.real
            elif isinstance(prediction, Rational):
                prediction = float(prediction)

        return prediction

    def compare_w_digits(self, reference: str, answer: str) -> bool:
        if reference.isdigit() and answer.isdigit():
            return abs(float(reference) - float(answer)) <= 0.05 * float(reference)
        else:
            return reference in answer

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task."""
        # Follow the implementation from TheoremQA
        # https://github.com/wenhuchen/TheoremQA/blob/123e36beaaa97c01f28a582f13c4f77a6822c199/predict_accuracy.py#L301C9-L317C1
        if not solution: # empty
            return False
        prediction = self.extract_answer(solution)
        if prediction is None or prediction == "":
            return False
        # LOGGER.info(f"GPQA Parsed Prediction: {prediction}")
        answer_type = self._answer_type
        gt = self._reference

        if isinstance(prediction, (str, int, float)) or isinstance(prediction, list):
            # Comparing prediction against the reference
            if answer_type in ["bool", "option", "Option"]:
                cur_correct = int(prediction == f"({gt})") or int(prediction == gt)
            elif answer_type == "integer":
                cur_correct = int(compare_two_numbers(prediction, gt))
            elif answer_type == "float":
                cur_correct = int(compare_two_numbers(prediction, gt))
            elif answer_type in ["list of integer", "list of float"]:
                cur_correct = int(compare_two_list(prediction, gt))
        else:
            cur_correct = 0
        return bool(cur_correct)


    def extract_options(self, prompt: str) -> dict:
        # Find the possible option separators (comma, semicolon, or parentheses)
        prompt = prompt.split("Options: ")[-1]
        # Extract the options using the delimiter
        options_match = prompt.split(" , ")
        options = {}
        for i in range(len(options_match)):
            option = options_match[i].strip("[]' ")
            option = option.split(")")
            letter = option[0].lower().strip()
            content = (
                option[1]
                .lower()
                .strip(".")
                .replace(". Which option is correct?", "")
                .replace(". Which one is correct?", "")
                .strip()
            )
            options.update({letter: content})
        return options

