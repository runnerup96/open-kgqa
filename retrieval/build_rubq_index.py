
import json
import os

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

    global_entities = json.load(open("/home/somov/open_kgqa/retrieval/index_dumps/entities_index/entities.json"))

    rubq_train = json.load(
        open('/home/somov/open_kgqa/data/RuBQ/RuBQ_2.0/train_with_aliases.json', 'r'))
    rubq_test = json.load(
        open('/home/somov/open_kgqa/data/RuBQ/RuBQ_2.0/test_with_aliases.json'))

    all_samples = rubq_train + rubq_test

    # add all enties from train/test to index
    rubq_entities_dict, _ = extract_entities_dict(all_samples, 'en')
    # get all test_entities for evaluation
    _, gold_entities_list = extract_entities_dict(rubq_test, 'en')

    list_for_dump = []
    for key, value in global_entities.items():
        d = {"id": key, "contents": value['label']}
        list_for_dump.append(d)

    for key, content in rubq_entities_dict.items():
        d = {"id": key, "contents": content}
        list_for_dump.append(d)

    if not os.path.exists("resources"):
        os.makedirs("resources")

    json.dump(list_for_dump, open('resources/global_entities_and_rubq_aliases_dump.json', 'w'),
              ensure_ascii=False, indent=4)