import pickle
import argparse

import training_utils
import preprocessing_utils

def preprocess_and_mask(query):
    preprocessed_sparql = preprocessing_utils.preprocess_sparql(query)
    masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)['masked_query']
    return masked_sparql


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_preds_pkl") # model inference output
    parser.add_argument("--path_to_gold_json") # model test part
    parser.add_argument("--dataset_name")
    args = parser.parse_args()

    preds_dict = pickle.load(open(args.path_to_preds_pkl, 'rb'))
    id2gold_query = dict()
    if args.dataset_name == 'rubq':
        kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_gold_json)
        id2gold_query = {sample['id']: preprocess_and_mask(sample['sparql']) for sample in kgqa_test_dataset_list}


    evaluator = training_utils.Evaluator()
    exact_match = 0

    for id_, gold_query in id2gold_query.items():
        pred_query = preds_dict[id_]['query']
        exact_match += evaluator.exact_match.compute(predictions=[pred_query], references=[gold_query],
                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')['exact_match']

    exact_match = round(exact_match / len(id2gold_query), 3)
    print('Exact match: ', exact_match)

