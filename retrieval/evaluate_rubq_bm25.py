
import json
import re
import os
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
import time

def extract_entities_dict(rubq_split, lang):
    result_dict = dict()
    ids_list = []
    for sample in rubq_split:
        id2alias = sample.get('id2alias')
        sample_id_list = []
        if id2alias:
            for key, value_dict in id2alias.items():
                lang_alias_list = value_dict.get(lang)
                if lang_alias_list:
                    entity_string = ", ".join(lang_alias_list)
                    result_dict[key] = entity_string
                    sample_id_list.append(key)
        ids_list.append(sample_id_list)
    return result_dict, ids_list

if __name__ == "__main__":

    rubq_test = json.load(
        open('/home/somov/open_kgqa/data/RuBQ/RuBQ_2.0/test_with_aliases.json'))

    _, gold_entities_list = extract_entities_dict(rubq_test, 'en')

    index_dir = "/home/somov/open_kgqa/retrieval/indexes/rubq_bm25_index_with_global"

    print('Initialize Lucene Index...')
    time_start = time.time()
    searcher = LuceneSearcher(index_dir)
    time_end = time.time()
    print(f'Index is created in {round(time_end - time_start, 3)}')

    k1, b = 0.3, 0.2
    searcher.set_bm25(k1, b)

    # go through rubq test
    # search top 10 candidates
    # evaluate precision/recall
    print('Begin evaluation...')
    overall_precision, overall_recall, overall_f1 = 0, 0, 0
    for sample, gold_ids in tqdm(zip(rubq_test, gold_entities_list), total=len(rubq_test)):
        query = sample['question_eng']
        result = searcher.search(query, k=100)

        pred_ids = []
        for result in result:
            doc_id = result.docid
            pred_ids.append(doc_id)

        true_positives = set(gold_ids) & set(pred_ids)

        precision = len(true_positives) / len(pred_ids) if pred_ids else 0.0

        # Recall: Proportion of gold entities that are correctly predicted
        recall = len(true_positives) / len(gold_ids) if gold_ids else 0.0

        # F1-Score: Harmonic mean of Precision and Recall
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        overall_precision += precision
        overall_recall += recall
        overall_f1 += f1
    print('Complete!')

    overall_precision /= len(rubq_test)
    overall_recall /= len(rubq_test)
    overall_f1 /= len(rubq_test)

    print('Precision: ', overall_precision)
    print('Recall: ', overall_recall)
    print('F1: ', overall_f1)


















