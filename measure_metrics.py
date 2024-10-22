
import pickle
import argparse
import training_utils
import preprocessing_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_preds_pkl")
    parser.add_argument("--path_to_gold_json")
    parser.add_argument("--dataset_name")

    args = parser.parse_args()

    preds_dict = pickle.load(open(args.path_to_preds_pkl, 'rb'))
    id2gold_query = dict()
    if args.dataset_name == 'rubq':
        kgqa_test_dataset_list = training_utils.format_rubq_dataset_to_kgqa_dataset(args.path_to_gold_json)
        id2gold_query = dict()
        for sample in kgqa_test_dataset_list:
            id_ = sample['id']
            sparql = sample['sparql']

            preprocessed_sparql = preprocessing_utils.preprocess_sparql(sparql)
            masked_sparql = preprocessing_utils.form_masked_query(preprocessed_sparql)['masked_query']
            id2gold_query[id_] = masked_sparql

    evaluator = training_utils.Evaluator()
    exact_match = 0

    for id_, gold_query in id2gold_query.items():
        pred_query = preds_dict[id_]['query']

        exact_match += evaluator.exact_match.compute(predictions=[pred_query], references=[gold_query],
                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')['exact_match']

    exact_match = round(exact_match / len(id2gold_query), 3)
    print('Exact match: ', exact_match)

    # TODO: Реализовать рассчет exec match для каждого датасета


