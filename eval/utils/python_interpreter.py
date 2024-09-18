from typing import Mapping
import re
import signal
from contextlib import contextmanager
from typing import Any
import subprocess
from tqdm import tqdm

import os
class PythonREPL():
    def __init__(self, timeout=5, tmp_file="cache/tmp"):
        self.timeout = timeout
        
        import datetime
        import random
        current_time = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        random_number = random.random()
        self.tmp_file = tmp_file + current_time + str(random_number)
        os.system(f"touch {self.tmp_file}.py" )

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm
 
    def __call__(self, query: str) -> str:
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)

        with open(f'{self.tmp_file}.py', "w") as f:
            f.write(query)
        
        with self.time_limit(self.timeout):
            result = subprocess.run(
                    ['python3', f'{self.tmp_file}.py'], capture_output=True, check=False, text=True, timeout=self.timeout)

            if result.returncode == 0:
                output = result.stdout
                return True, output.strip()
            else:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif self.tmp_file in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(f'/{self.tmp_file}.py') + 1 if f'/{self.tmp_file}.py' in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
        
    
def postprocess_completion(executor, completion):

    executions = ["!" + code for code in re.findall(r"```bash(.*?)```", completion, re.DOTALL) if "!" not in code]
    executions.extend(re.findall(r"```python(.*?)```", completion, re.DOTALL))
    executions.extend(re.findall(r"<execute>(.*?)</execute>", completion, re.DOTALL))
    
    if len(executions) == 0: # directly return cot result
        return completion
    else:
        ### Python
        execution_outputs = []
        for code in executions:
            try: 
                success, output = executor(code)
            except TimeoutError:
                print("time out")
                # success = False
                output = ""
            else:
                output = output if success else ""
            execution_outputs.append(output)
        extracted_outputs = execution_outputs

        for index in range(1, len(extracted_outputs) + 1):
            extracted_solution = str(extracted_outputs[-index]).strip()
            break

        return extracted_solution


# def postprocess_completions(completion_list):
#     executor = PythonREPL()
    
#     solution_list = []
#     for completion in completion_list:
#         solution_list.append(postprocess_completion(executor, completion))

#     del executor

#     return solution_list


import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def postprocess_completion_wrapper(completion):
    executor = PythonREPL()
    result = postprocess_completion(executor, completion)
    os.system(f"rm {executor.tmp_file}.py")
    del executor
    return result

def postprocess_completions(completion_list):
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(postprocess_completion_wrapper, completion) for completion in completion_list]
        solution_list = [future.result() for future in as_completed(futures)]
    return solution_list

if __name__ == "__main__":
    code = """
Step 1: First, let's calculate the total number of eggs laid by Janet's ducks in a day.
Step 2: Next, let's calculate the number of eggs Janet eats for breakfast each day.
Step 3: Then, let's calculate the number of eggs Janet bakes for her friends each day.
Step 4: Finally, let's calculate the number of eggs Janet sells at the farmers' market each day.
Step 5: To find the total amount of money Janet makes each day at the farmers' market, we can multiply the number of eggs she sells by the price per egg.
```python
# Step 6: Calculate the total number of eggs laid by Janet's ducks in a day.
total_eggs_per_day = 16
# Step 7: Calculate the number of eggs Janet eats for breakfast each day.
eggs_eaten_per_day = 3
# Step 8: Calculate the number of eggs Janet bakes for her friends each day.
eggs_baked_per_day = 4
# Step 9: Calculate the number of eggs Janet sells at the farmers' market each day.
eggs_sold_per_day = total_eggs_per_day - eggs_eaten_per_day - eggs_baked_per_day
# Step 10: Calculate the total amount of money Janet makes each day at the farmers' market.
price_per_egg = 2
total_money_per_day = eggs_sold_per_day * price_per_egg
total_money_per_day
```
Answer:
12

"""
    import pandas as pd
    import json
    code_list = pd.read_json("../Math/math/output_llama3-8b-new-mix.json").to_dict("records")
    completions = [code["completions"] for code in code_list]
    processed_completions = postprocess_completions(completions[:10])
    print(completions[:10])
    print(processed_completions[:10])