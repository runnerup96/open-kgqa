#!/bin/bash

# only local run!
export PYTHONPATH='/Users/20652092/Desktop/open-kgqa'
export PROJECT_PATH='/Users/20652092/Desktop/open-kgqa'

python_path='/usr/local/bin/python3.10'

test_path="/Users/20652092/Desktop/open-kgqa/data/RuBQ_2.0/test.json"
preds_path="/Users/20652092/Desktop/open-kgqa/experiments/fred_t5_xxl_rubq_with_preds_s2/rubq_test_inference_result_kgqa_full.json"
dataset_name="rubq"

$python_path -u measure_exec_match.py \
  --path_to_preds_json $preds_path \
  --path_to_gold_json $test_path \
  --dataset_name $dataset_name


# проверить что в кеше нет грязи
# проверить метрику и что в резах мб грузяь это норм