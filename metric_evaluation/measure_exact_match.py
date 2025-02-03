import os
import pickle
import argparse

import training_utils
import preprocessing_utils
import json

def preprocess_sparql(query):
    preprocessed_sparql = preprocessing_utils.preprocess_sparql(query)
    # masked_sparql = preprocessing_utils.replace_preds(preprocessed_sparql)
    # masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)['masked_query']
    return preprocessed_sparql


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
        id2gold_query = {sample['id']: preprocess_sparql(sample['sparql']) for sample in kgqa_test_dataset_list}

    kgqa_test_dataset_list = json.load(open(f"clean_data/preprocessed/{args.dataset_name}/{args.dataset_name}_test.json"))['dataset']
    id2gold_query = {sample['id']: preprocess_sparql(sample['query']) for sample in kgqa_test_dataset_list}

    relations_description = json.load(open("clean_data/wikidata/relations.json", 'r'))


    evaluator = training_utils.Evaluator()
    exact_match = 0

    failed_pairs = []
    pairs = []
    overall_exact_match = 0
    for id_, gold_query in id2gold_query.items():
        pred_query = preds_dict[str(id_)]['query']
        # TODO: Revert masking
        # select ?answer where { wd:Q173985 country ?answer } -> select ?answer where { wd:Q173985 wdt:P123 ?answer }

        # replaced_gold_query = preprocessing_utils.replace_preds(gold_query, relations_description)
        test_replacement_dict = json.load(open(f"clean_data/sft/{args.dataset_name}_test_replacement_dict.json", 'r'))
        reverted_sparql = preprocessing_utils.replace_labels_in_sparql(pred_query, test_replacement_dict.get(str(id_)))

        pairs.append([gold_query, reverted_sparql])

        print(gold_query)
        print(reverted_sparql)
        print(test_replacement_dict.get(str(id_)))
        print()

        exact_match = evaluator.exact_match.compute(predictions=[reverted_sparql], references=[gold_query],
                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')['exact_match']

        overall_exact_match += exact_match

        if exact_match == 0:
            failed_pairs.append([gold_query, gold_query])


    json.dump(pairs,
              open(f"clean_data/sft/{args.dataset_name}_pairs.json", 'w'),
              ensure_ascii=False, indent=4)

    exact_match = round(overall_exact_match / len(id2gold_query), 3)
    print('Exact match: ', exact_match)

