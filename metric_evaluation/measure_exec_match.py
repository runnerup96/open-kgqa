import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import time

import training_utils
import preprocessing_utils
from wikidata_connector import BlazegraphConnector


if __name__ == "__main__":
    def check_gold_answer(answers: list):
        if answers:
            if not any([x.startswith("Exception:") for x in answers]):
                return True
        return False

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_preds_json") # kgqa inference output [{id, filled_query}, ...]
    parser.add_argument("--path_to_gold_json")  # model test part [{id, sparql}, ...]
    parser.add_argument("--dataset_name")
    parser.add_argument("--cache_file_path",
                        default=os.path.join(os.environ["PROJECT_PATH"], "metric_evaluation/_gold_answers_cached.json"))
    args = parser.parse_args()

    bgc = BlazegraphConnector()
    cache_file = json.load(open(args.cache_file_path, 'r')) if os.path.exists(args.cache_file_path) else dict()


    # make id2gold_q
    if args.dataset_name == 'rubq':
        kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_gold_json)
        id2gold_query = {sample['id']: sample['sparql'] for sample in kgqa_test_dataset_list}

    # get id2gold_a and upd cache
    id2gold_answer = dict()
    for id_, gold_query in tqdm(id2gold_query.items()):
        gold_answer = cache_file[id_] if id_ in cache_file else bgc.fetch_query(gold_query)
        id2gold_answer[id_] = gold_answer
        if check_gold_answer(gold_answer):
            cache_file[id_] = gold_answer
    json.dump(cache_file, open(args.cache_file_path, 'w', encoding='utf-8'), ensure_ascii=False)


    # make id2pred_q
    preds = json.load(open(args.path_to_preds_json, 'rb'))
    id2pred_query = {dic['id']: dic['filled_query'] for dic in preds}
    assert set(id2gold_query.keys()) == set(id2pred_query.keys())

    # get id2pred_a without cache
    id2pred_answer = dict()
    for id_, pred_query in tqdm(id2pred_query.items()):
        pred_answer = bgc.fetch_query(pred_query)
        id2pred_answer[id_] = pred_answer


    # check only correct gold cases
    exec_match = round(np.mean(
        [id2gold_answer[id_] == id2pred_answer[id_] for id_ in id2gold_answer.keys()
         if check_gold_answer(id2gold_answer[id_])]
    ), 3)

    exec_mathc_result = {
        id_: {
            'gold_query': id2gold_query[id_],
            'gold_answer': id2gold_answer[id_],
            'pred_query': id2pred_query[id_],
            'pred_answer': id2pred_answer[id_]
        }
        for id_ in id2gold_answer.keys()
    }
    exec_mathc_result = {**{'exec_match (correct golds)': exec_match}, **exec_mathc_result}
    print('Exec match: ', exec_match)

    output_dir = os.path.dirname(args.path_to_preds_json)
    filename, _ = os.path.splitext(os.path.basename(args.path_to_preds_json))
    save_path = os.path.join(output_dir, 'exec_match_results.json')
    json.dump(exec_mathc_result, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print('Results save at preds path')