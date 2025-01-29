import json
import preprocessing_utils
import prompts
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def write_tsv(examples, filename, expected_num_columns=2):
    """Write examples to tsv file."""
    with open(filename, "w") as tsv_file:
        for example in examples:
            if len(example) != expected_num_columns:
                raise ValueError("Example '%s' has %s columns." %
                                 (example, len(example)))
            example = "\t".join(example)
            line = "%s\n" % example
            tsv_file.write(line)
    print("Wrote %s examples to %s." % (len(examples), filename))


def get_alias_list(alias_lang_dict):
    alias_list = []
    if "en" in alias_lang_dict:
        alias_list = alias_lang_dict["en"]
    return alias_list


def format_id2alias(id2alias):
    alias_string_list = []
    for wikidata_id, alias_lang_dict in id2alias.items():
        alias_list = get_alias_list(alias_lang_dict)[:5]
        alias_string = ", ".join(alias_list)
        alias_string_list.append(f"{wikidata_id}: {alias_string}")
    alias_string = ", ".join(alias_string_list)
    return alias_string


def format_sft_sample(question, alias_string, graph_preds, graph_namespaces):
    # sft_prompt = f"""
    #             QUESTION: {question}
    #             QUESTION ENTITIES: {alias_string}
    #             KNOWLEDGE GRAPH NAMESPACES: {graph_namespaces}
    #             KNOWLEDGE GRAPH PREDICATES: {graph_preds}
    #             """
    sft_prompt = f"""
                    QUESTION: {question}
                    QUESTION ENTITIES: {alias_string}
                    """
    return sft_prompt


def format_source_dataset(rubq_dataset, phase):
    sft_examples_list, failed_samples = [], []

    for sample in tqdm(rubq_dataset, total=len(rubq_dataset)):

        sample_id = str(sample['uid'])
        if language == 'ru':
            question = sample['question_ru']
            instruction = prompts.RU_INSTRUCTION
        else:
            question = sample['question_eng']
            instruction = prompts.EN_INSTRUCTION

        id2alias = sample['id2alias']
        sparql = sample['query']
        if sparql and id2alias:
            sparql = preprocessing_utils.preprocess_sparql(sparql)

            #replace preds with
            sparql = preprocessing_utils.replace_preds(sparql, relations_description)


            alias_string = format_id2alias(id2alias)
            if alias_string:
                user_task = format_sft_sample(question, alias_string, knowledge_graph_predicates,
                                              knowledge_graph_namespaces)
                if phase == 'train':
                    chat = [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_task},
                        {"role": "assistant", "content": sparql}
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                else:
                    chat = [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_task}
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                sft_examples_list.append({"id": sample_id, "sft": formatted_prompt})
            else:
                failed_samples.append(sample)
        else:
            failed_samples.append(sample)

    return sft_examples_list, failed_samples


if __name__ == "__main__":

    rubq_train = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/train_with_aliases.json', 'r'))
    rubq_test = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/test_with_aliases.json', 'r'))

    language = 'en'

    predicate_description_dict = json.load(open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_predicate_mapping.json", 'r'))
    knowledge_graph_predicates = ", ".join(list(predicate_description_dict.keys()))
    knowledge_graph_namespaces = ", ".join(['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'])

    # TODO: Нужно собрать полный словарь всех придикатов и квалификаторов
    relations_description = json.load(open("/Users/somov-od/Documents/phd/projects/open_kgqa/dataset_managers/relations.json", 'r'))

    predicate_description_dict = json.load(open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_predicate_mapping.json", 'r'))

    phase = 'train'

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", use_fast=True)
    train_sft_examples_list, train_failed_samples = format_source_dataset(rubq_train, 'train')
    valid_sft_examples_list, valid_failed_samples = format_source_dataset(rubq_test, 'train')
    train_for_test_sft_examples_list, train_for_test_failed_samples = format_source_dataset(rubq_train, 'test')
    test_sft_examples_list, test_failed_samples = format_source_dataset(rubq_test, 'test')


    json.dump(train_sft_examples_list, open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_sft_train_no_preds.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(train_for_test_sft_examples_list,
              open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_sft_train_for_test.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(valid_sft_examples_list,
              open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_sft_valid_no_preds.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(test_sft_examples_list,
              open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_sft_test_no_preds.json", 'w'),
              ensure_ascii=False, indent=4)


    print('Prepared SFT train samples: ', len(train_sft_examples_list))
    print('Total of train failed samples: ', len(train_failed_samples))

    print('Prepared SFT train samples: ', len(test_sft_examples_list))
    print('Total of test failed samples: ', len(test_failed_samples))









