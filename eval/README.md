# EVAL

## Coding

### human_eval

```bash
cd Coding/human_eval
python evaluate_human_eval_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --save_dir ./ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.2
```

### leetcode

```bash
cd Coding/leetcode
python evaluate_leetcode_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --save_dir ./ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.
```

### mbpp

```bash
cd Coding/mbpp
python run_mbpp_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --input_data 	new_mbpp.json \
  --save_dir ./ \
  --model_type mistral 
```

## Math

### math

```bash
cd Math/math
python evaluate_math_chat_quicktest.py \
  --data_dir ./ \
  --save_dir ./ \
  --model_type mistral \
  --model ../../../models/eurus-7b-kto 
```



```bash
cd Math/math
python evaluate_math_ui_quicktest.py \
  --data_dir ./ \
  --save_dir ./ \
  --model_type mistral \
  --model ../../../models/eurus-7b-kto 
```

### theorem_qa

```bash
cd Math/theorem_qa
python theorem_qa_ui_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir ./
```



```bash
cd Math/theorem_qa
python theorem_qa_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir ./
```

### SVAMP&ASDiv&GSM-Plus

```bash
cd Math/subset
python subset.py \
  --data_dir ./data \
  --save_dir ./result \
  --model_type mistral \
  --model ../../../models/eurus-7b-kto 
```



```bash
cd Math/subset
python subset_ui_quicktest.py \
  --data_dir ./data \
  --save_dir ./ui_results \
  --model_type mistral \
  --model ../../../models/eurus-7b-kto 
```



## Reasoning

### BBH

```bash
cd Reasoning/bbh
python run_bbh_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --data_filepath ./test_prompts.json \
  --output_filepath ./res.jsonl \
  --model_type mistral \
  --n_processes 8
```



## Ins-Following

### if_eval

```bash
cd Ins-Following/if_eval
python generate_response_chat_quicktest.py \
  --model ../../../models/eurus-7b-kto \
  --input_data 	./input_data.jsonl \
  --save_path ./input_response_data.jsonl \
  --model_type mistral 
```

