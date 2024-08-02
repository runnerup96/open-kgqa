#!/bin/bash


CUDA_DEVICE_NUMBER='1'
seed=2


epoch=150
train_batch_size=8
gradient_accumulation_steps=32
eval_batch_size=8

input_length=1024
output_length=140
num_beams=1

lr='1e-3'
project_dir="/home/somov/open_kgqa"

#dataset_name="rubq"
dataset_name="salute"

# rubq
#epoch=150
# salute
epoch=2


language="ru"

#data_path="data/RuBQ/RuBQ_2.0"
data_path="data/Salute"

save_model_dir="experiments"

model_name="ai-forever/FRED-T5-1.7B"
dir_model_name="fred_t5_xxl"
run_explain_name="with_preds"

log_steps=5
eval_steps=100

train_file="$project_dir/$data_path/train.json"
test_file="$project_dir/$data_path/test.json"

#predicate_mapping="$project_dir/$data_path/rubq_predicate_mapping.json"
predicate_mapping="$project_dir/$data_path/kgqa_query_vocab.json"

run_name="${dir_model_name}_${dataset_name}_${run_explain_name}_s$seed"
output_dir="$project_dir/$save_model_dir/$run_name"
logs_dir="$output_dir/training_logs"


tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python -u hf_t5_modeling.py \
                            --model_name_or_path $model_name \
                            --sparql_dataset_name $dataset_name \
                            --language $language \
                            --path_to_training_file $train_file \
                            --path_to_testing_file $test_file \
                            --path_to_predicate_description $predicate_mapping \
                            --do_train \
                            --do_eval \
                            --predict_with_generate \
                            --learning_rate $lr \
                            --max_grad_norm 1.0 \
                            --seed $seed \
                            --per_device_train_batch_size $train_batch_size \
                            --per_device_eval_batch_size $eval_batch_size \
                            --gradient_accumulation_steps $gradient_accumulation_steps \
                            --num_train_epochs $epoch \
                            --max_seq_length $input_length  \
                            --max_output_length $output_length \
                            --save_strategy 'steps' \
                            --evaluation_strategy 'steps' \
                            --metric_for_best_model 'eval_exact_match' \
                            --load_best_model_at_end \
                            --eval_delay $eval_steps \
                            --eval_steps $eval_steps \
                            --save_steps $eval_steps \
                            --eval_accumulation_steps $gradient_accumulation_steps \
                            --num_beams 1 \
                            --logging_steps $log_steps \
                            --report_to 'tensorboard' \
                            --save_total_limit 1 \
                            --overwrite_output_dir \
                            --output_dir $output_dir \
                            --logging_dir $logs_dir \
                            --phase 'original'" ENTER


#output_dir="/home/somov/naacl_cp_t5/experiments/mt0-base_ml_pauq_xsp_s42"
#test_file="/home/somov/naacl_cp_t5/data/prepared_data/ru_pauq_xsp/ru_pauq_xsp_test.tsv"
#eval_batch_size=128
#run_name="eval_on_ru_pauq_ml"
#tmux new-session -d -s $run_name
#

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python -u hf_t5_inference.py \
                              --model_name_or_path $output_dir \
                              --sparql_dataset_name $dataset_name \
                              --language $language \
                              --path_to_testing_file $test_file \
                              --path_to_predicate_description $predicate_mapping \
                              --seed $seed \
                              --max_seq_length $input_length  \
                              --max_output_length $output_length \
                              --per_device_eval_batch_size $eval_batch_size \
                              --eval_accumulation_steps $gradient_accumulation_steps \
                              --num_beams $num_beams \
                              --output_dir $output_dir" ENTER

tmux a -t $run_name



