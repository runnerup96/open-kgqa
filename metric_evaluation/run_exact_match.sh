#!/bin/bash

export PYTHONPATH='/raid/home/msbutko/training/open_kgqa'
export PROJECT_PATH='/raid/home/msbutko/training/open_kgqa'

python_path='/raid/home/msbutko/env_open_kgqa/bin/python3.9'

test_path="/raid/home/msbutko/training/open_kgqa/data/RuBQ_2.0/test.json"
preds_path="/raid/home/msbutko/training/open_kgqa/experiments/fred_t5_xxl_rubq_with_preds_s2/rubq_test_inference_result.pkl"
dataset_name="rubq"

$python_path -u measure_exact_match.py \
  --path_to_preds_pkl $preds_path \
  --path_to_gold_json $test_path \
  --dataset_name $dataset_name