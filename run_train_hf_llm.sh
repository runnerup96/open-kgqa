#!/bin/bash


CUDA_DEVICE_NUMBER='1'
seed=1

#10
llama3_model_path="/home/somov/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
#144
#llama3_model_path="/home/somov/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"


project_path="/home/somov/open_kgqa"
save_model_dir="experiments"
dataset_name="rubq"
language="ru"
data_path="data/RuBQ/RuBQ_2.0"

train_file="$project_path/$data_path/train.json"
test_file="$project_path/$data_path/test.json"
predicate_mapping="$project_path/$data_path/rubq_predicate_mapping.json"

run_explain_name="lora"
output_dir="$project_path/$save_model_dir/${dataset_name}_s${seed}_${run_explain_name}"
run_name="${dataset_name}_s${seed}_${run_explain_name}"

log_dir="$output_dir/training_logs"


#lora
train_batch_size=4
eval_batch_size=8
gradient_accumulation_steps=24
eval_accumulation_steps=4
lr="1.5e-4"

#sft
#train_batch_size=1
#eval_batch_size=1
#gradient_accumulation_steps=48
#eval_accumulation_steps=48
#lr="1e-5"

input_seq_length=256
output_seq_length=140
num_beans=1

epochs_number=20

tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python3 hf_llm_modeling.py \
    --model_name_or_path $llama3_model_path \
    --use_lora \
    --sparql_dataset_name $dataset_name \
    --language $language \
    --path_to_training_file $train_file \
    --path_to_testing_file $test_file \
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
    --num_train_epochs $epochs_number \
    --run_name $run_name" ENTER

trained_model_path="$output_dir/final_checkpoints"
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python3 hf_llm_inference.py \
    --model_name_or_path $trained_model_path \
    --use_lora \
    --sparql_dataset_name $dataset_name \
    --language $language \
    --path_to_testing_file $test_file \
    --path_to_predicate_description $predicate_mapping \
    --seed $seed \
    --max_seq_length $input_seq_length \
    --max_new_tokens $output_seq_length \
    --per_device_eval_batch_size $eval_batch_size \
    --num_beams $num_beans \
    --output_dir $output_dir" ENTER

tmux a -t $run_name
