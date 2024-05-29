MODEL_DIR=$1
if [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="/data/sbj/eurus-70b-sft"
fi
MODEL_NAME=$(basename "$MODEL_DIR")
OUTPUT_DIR="/data/checkpoints/results/$MODEL_NAME" 
mkdir -p $OUTPUT_DIR
mkdir -p cache

# coding_human_eval
echo "running human_eval evaluation"
mkdir -p $OUTPUT_DIR/human_eval
cd Coding/human_eval
python3 evaluate_human_eval.py \
  --model $MODEL_DIR \
  --save_dir $OUTPUT_DIR/human_eval/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.2
cd ../..


# coding_leetcode
echo "running leetcode evaluation"
mkdir -p $OUTPUT_DIR/leetcode
cd Coding/leetcode
python3 evaluate_leetcode.py \
  --model $MODEL_DIR \
  --save_dir $OUTPUT_DIR/leetcode/ \
  --num-samples-per-task 1 \
  --model_type mistral \
  --temperature 0.

python3 test.py \
  --generation_path $OUTPUT_DIR/leetcode/ \
  --result_path $OUTPUT_DIR/leetcode/ \
  --temp_dir output/temp
cd ../..

# coding_mbpp
echo "running mbpp evaluation"
mkdir -p $OUTPUT_DIR/mbpp
cd Coding/mbpp
python3 evaluate_mbpp.py \
  --model $MODEL_DIR \
  --input_data 	new_mbpp.json \
  --save_dir $OUTPUT_DIR/mbpp/ \
  --model_type mistral 
cd ../..


# math_math
echo "running math-cot evaluation"
mkdir -p $OUTPUT_DIR/math_cot
cd Math/math
python3 evaluate_math_cot.py \
  --data_dir ./ \
  --save_dir $OUTPUT_DIR/math_cot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

echo "running math-pot evaluation"
mkdir -p $OUTPUT_DIR/math_pot
cd Math/math
mkdir -p cache
python3 evaluate_math_pot.py \
  --data_dir ./ \
  --save_dir $OUTPUT_DIR/math_pot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

# math_asdiv_gsmplus_svamp
echo "running asdiv&gsmplus&svamp cot evaluation"
mkdir -p $OUTPUT_DIR/subset_cot
cd Math/subset
python3 evaluate_subset_cot.py \
  --data_dir ./data \
  --save_dir $OUTPUT_DIR/subset_cot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

echo "running asdiv&gsmplus&svamp pot evaluation"
mkdir -p $OUTPUT_DIR/subset_pot
cd Math/subset
mkdir -p cache
python3 evaluate_subset_pot.py \
  --data_dir ./data \
  --save_dir $OUTPUT_DIR/subset_pot/ \
  --model_type mistral \
  --model $MODEL_DIR 
cd ../..

# math_theorem_qa
echo "running theorem-qa cot evaluation"
mkdir -p $OUTPUT_DIR/theorem_qa_cot
cd Math/theorem_qa
python3 evaluate_theorem_qa_cot.py \
  --model $MODEL_DIR \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir $OUTPUT_DIR/theorem_qa_cot/
cd ../..

echo "running theorem-qa pot evaluation"
mkdir -p $OUTPUT_DIR/theorem_qa_pot
cd Math/theorem_qa
mkdir -p cache
python3 evaluate_theorem_qa_pot.py \
  --model $MODEL_DIR \
  --input_data 	./theorem_qa.json \
  --model_type mistral \
  --save_dir $OUTPUT_DIR/theorem_qa_pot/
cd ../..

# reasoning_bbh
echo "running bbh evaluation"
mkdir -p $OUTPUT_DIR/bbh
cd Reasoning/bbh
python3 evaluate_bbh.py \
  --model $MODEL_DIR \
  --data_filepath ./test_prompts.json \
  --output_filepath $OUTPUT_DIR/bbh/res.jsonl \
  --model_type mistral \
  --n_processes 8
cd ../..

# ins-Following-if_eval
echo "running if-eval evaluation"
mkdir -p $OUTPUT_DIR/if_eval
cd Ins-Following/if_eval
python3 evaluate_if_eval.py \
  --model $MODEL_DIR \
  --input_data ./input_data.jsonl \
  --save_path $OUTPUT_DIR/if_eval/input_response_data.jsonl \
  --model_type mistral 

python3 evaluation_main.py \
  --input_data ./input_data.jsonl \
  --input_response_data $OUTPUT_DIR/if_eval/input_response_data.jsonl \
  --output_dir $OUTPUT_DIR/if_eval/
cd ../..

#mmlu
echo "running mmlu evaluation"
mkdir -p $OUTPUT_DIR/mmlu
cd mmlu/
python3 -u evaluate_mmlu.py \
    --model $MODEL_DIR \
    --data_dir ./ \
    --save_dir $OUTPUT_DIR/mmlu



