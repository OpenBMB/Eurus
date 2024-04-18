MODEL_DIR="/data/cgq/models/eurus-7b-kto" 


mkdir cache
mkdir result
mkdir result/human_eval
mkdir result/leetcode
mkdir result/mbpp
mkdir result/math_cot
mkdir result/math_pot
mkdir result/theorem_qa_cot
mkdir result/theorem_qa_pot
mkdir result/subset_cot
mkdir result/subset_pot
mkdir result/bbh
mkdir result/if_eval

# coding_human_eval
echo "running human eval evaluation"
cd Coding/human_eval
python evaluate_human_eval.py \
  --model $MODEL_DIR \
  --save_dir ../../result/human_eval/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.2
cd ../..

# coding_leetcode
echo "running leetcode evaluation"
cd Coding/leetcode
python evaluate_leetcode.py \
  --model $MODEL_DIR \
  --save_dir ../../result/leetcode/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.
cd ../..

# coding_mbpp
echo "running mbpp evaluation"
cd Coding/mbpp
python evaluate_mbpp.py \
  --model $MODEL_DIR \
  --input_data 	new_mbpp.json \
  --save_dir ../../result/mbpp/
  --model_type mistral \
cd ../..


# math_math
echo "running math-cot evaluation"
cd Math/math
python evaluate_math_cot.py \
  --data_dir ./ \
  --save_dir ../../result/math_cot/ \
  --model_type mistral \
  --model $MODEL_DIR 

echo "running math-pot evaluation"
python evaluate_math_pot.py \
  --data_dir ./ \
  --save_dir ../../result/math_pot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

# math_asdiv_gsmplus_svamp
echo "running asdiv&gsmplus&svamp cot evaluation"
cd Math/subset
python evaluate_subset_cot.py \
  --data_dir ./data \
  --save_dir ../../result/subset_cot/ \
  --model_type mistral \
  --model $MODEL_DIR 

echo "running asdiv&gsmplus&svamp pot evaluation"
python evaluate_subset_pot.py \
  --data_dir ./data \
  --save_dir ../../result/subset_pot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

# math_theorem_qa
echo "running theorem-qa cot evaluation"
cd Math/theorem_qa
python evaluate_theorem_qa_cot.py \
  --model $MODEL_DIR \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir ../../result/theorem_qa_cot/


echo "running theorem-qa pot evaluation"
python evaluate_theorem_qa_pot.py \
  --model $MODEL_DIR \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir ../../result/theorem_qa_pot/
cd ../..

# reasoning_bbh
echo "running bbh evaluation"
cd Reasoning/bbh
python evaluate_bbh.py \
  --model $MODEL_DIR \
  --data_filepath ./test_prompts.json \
  --output_filepath ../../result/bbh/res.jsonl \
  --model_type mistral \
  --n_processes 8
cd ../..

# ins-Following-if_eval
echo "running if-eval evaluation"
cd Ins-Following/if_eval
python evaluate_if_eval.py \
  --model $MODEL_DIR \
  --input_data ./input_data.jsonl \
  --save_path ../../result/if_eval/input_response_data.jsonl \
  --model_type mistral 

python evaluation_main.py \
  --input_data ./input_data.jsonl \
  --input_response_data ../../result/if_eval/input_response_data.jsonl \
  --output_dir ../../result/if_eval/

cd ../..



