import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import time
import pickle

import training_utils
import preprocessing_utils
from wikidata_connector import BlazegraphConnector

def preprocess_sparql(query):
    preprocessed_sparql = preprocessing_utils.preprocess_sparql(query)
    # masked_sparql = preprocessing_utils.replace_preds(preprocessed_sparql)
    # masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)['masked_query']
    return preprocessed_sparql

if __name__ == "__main__":
    def check_gold_answer(answers: list):
        if answers:
            if not any([x.startswith("Exception:") for x in answers]):
                return True
        return False

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_preds_pkl")
    parser.add_argument("--path_to_gold_json")  # model test part [{id, sparql}, ...]
    parser.add_argument("--dataset_name")
    parser.add_argument("--cache_file_path",
                        default=os.path.join(os.environ["PROJECT_PATH"], "metric_evaluation/_gold_answers_cached.json"))
    args = parser.parse_args()

    bgc = BlazegraphConnector()
    cache_file = json.load(open(args.cache_file_path, 'r')) if os.path.exists(args.cache_file_path) else dict()

    kgqa_test_dataset_list = json.load(open(f"clean_data/preprocessed/{args.dataset_name}/{args.dataset_name}_test.json"))['dataset']
    id2gold_query = {sample['id']: preprocess_sparql(sample['query']) for sample in kgqa_test_dataset_list}
    test_replacement_dict = json.load(open(f"clean_data/sft/{args.dataset_name}_test_replacement_dict.json", 'r'))

    # make id2gold_q
    # if args.dataset_name == 'rubq':
    #     kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_gold_json)
    #     id2gold_query = {sample['id']: sample['sparql'] for sample in kgqa_test_dataset_list}

    # get id2gold_a and upd cache
    id2gold_answer = dict()
    for id_, gold_query in tqdm(id2gold_query.items()):
        gold_answer = cache_file[str(id_)] if str(id_) in cache_file else bgc.fetch_query(gold_query)
        id2gold_answer[str(id_)] = gold_answer
        if check_gold_answer(gold_answer):
            cache_file[str(id_)] = gold_answer
    json.dump(cache_file, open(args.cache_file_path, 'w', encoding='utf-8'), ensure_ascii=False)


    # make id2pred_q
    # preds = json.load(open(args.path_to_preds_json, 'rb'))
    preds = pickle.load(open(args.path_to_preds_pkl, 'rb'))
    # id2pred_query = {dic['id']: dic['query'] for dic in preds}
    id2pred_query = {str(id_): dic['query'] for id_, dic in preds.items()}
    id2gold_query = {str(k): v for k,v in id2gold_query.items()}

    assert set(id2gold_query.keys()) == set(id2pred_query.keys())

    # get id2pred_a without cache
    id2pred_answer = dict()
    for id_, pred_query in tqdm(id2pred_query.items()):
        reverted_sparql = preprocessing_utils.replace_labels_in_sparql(pred_query, test_replacement_dict.get(str(id_)))
        pred_answer = bgc.fetch_query(reverted_sparql)
        id2pred_answer[id_] = pred_answer


    # check only correct gold cases
    exec_match_arr = [set(id2gold_answer[id_]) == set(id2pred_answer[id_])
                      for id_ in id2gold_answer.keys()
                      if check_gold_answer(id2gold_answer[id_])]
    exec_match = round(np.mean(exec_match_arr), 3)

    exec_mathc_result = {
        **{'exec_match': {'metric': exec_match,
                          'correct': sum(exec_match_arr),
                          'total_gold': len(exec_match_arr)}},
        **{id_: {'gold_query': id2gold_query[id_],
                 'gold_answer': id2gold_answer[id_],
                 'pred_query': id2pred_query[id_],
                 'pred_answer': id2pred_answer[id_]}
           for id_ in id2gold_answer.keys()}
    }
    print(f'Exec match = {sum(exec_match_arr)} / {len(exec_match_arr)} = {exec_match}')

    output_dir = os.path.dirname(args.path_to_preds_json)
    filename, _ = os.path.splitext(os.path.basename(args.path_to_preds_json))
    save_path = os.path.join(output_dir, 'exec_match_results.json')
    json.dump(exec_mathc_result, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print('Results save at preds path')