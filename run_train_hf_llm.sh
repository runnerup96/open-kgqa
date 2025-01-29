#!/bin/bash


CUDA_DEVICE_NUMBER='0'
seed=42

model_name_or_path="Qwen/Qwen2.5-Coder-0.5B-Instruct"

project_path="/home/somov/open_kgqa"
save_model_dir="experiments"

dataset_name="rubq"

language="en"

data_path="data/RuBQ/RuBQ_2.0"

train_file="$project_path/$data_path/rubq_sft_train_no_preds.json"
valid_file="$project_path/$data_path/rubq_sft_valid_no_preds.json"
test_file="$project_path/$data_path/rubq_sft_test_no_preds.json"

predicate_mapping="$project_path/$data_path/rubq_predicate_mapping.json"

run_explain_name="kgqa_total_recall_sft_no_preds"
output_dir="$project_path/$save_model_dir/${dataset_name}_s${seed}_${run_explain_name}"
run_name="${dataset_name}_s${seed}_${run_explain_name}"

log_dir="$output_dir/training_logs"


#lora
#train_batch_size=16
#eval_batch_size=16
#gradient_accumulation_steps=8
#eval_accumulation_steps=1
#lr="1.5e-4"

#sft
train_batch_size=16
eval_batch_size=16
gradient_accumulation_steps=8
eval_accumulation_steps=8
lr="2e-4"

input_seq_length=1024
output_seq_length=140
num_beams=1

epochs_number=10
eval_steps=40
logging_steps=10

python_conda_path="/home/somov/miniconda3/envs/llm_tuning/bin/python"

tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' $python_conda_path hf_llm_modeling.py \
    --model_name_or_path $model_name_or_path \
    --sparql_dataset_name $dataset_name \
    --language $language \
    --path_to_training_file $train_file \
    --path_to_testing_file $valid_file \
    --path_to_predicate_description $predicate_mapping \
    --learning_rate $lr \
    --seed $seed \
    --per_device_train_batch_size $train_batch_size \
    --per_device_eval_batch_size $eval_accumulation_steps \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_seq_length $input_seq_length \
    --report_to 'tensorboard' \
    --overwrite_output_dir \
    --output_dir $output_dir \
    --logging_dir $log_dir \
    --save_steps $eval_steps \
    --logging_steps $logging_steps \
    --eval_steps $eval_steps \
    --num_train_epochs $epochs_number \
    --run_name $run_name" ENTER

trained_model_path="$output_dir/final_checkpoints"
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' $python_conda_path hf_llm_inference.py \
    --model_name_or_path $trained_model_path \
    --sparql_dataset_name $dataset_name \
    --language $language \
    --path_to_testing_file $test_file \
    --path_to_predicate_description $predicate_mapping \
    --seed $seed \
    --max_seq_length $input_seq_length \
    --max_new_tokens $output_seq_length \
    --per_device_eval_batch_size $eval_batch_size \
    --num_beams $num_beams \
    --output_dir $output_dir" ENTER

tmux a -t $run_name
