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


def choose_language(alias_lang_dict):
    if "en" in alias_lang_dict:
        alias_list = alias_lang_dict["en"]
    else:
        if "ru" in alias_lang_dict:
            alias_list = alias_lang_dict["ru"]
        else:
            random_lang = list(alias_lang_dict.keys())[0]
            alias_list = alias_lang_dict[random_lang]
    return alias_list


def format_id2alias(id2alias, lang):
    alias_string_list = []
    for wikidata_id, alias_lang_dict in id2alias.items():
        alias_list = choose_language(alias_lang_dict)
        alias_string = ", ".join(alias_list)
        alias_string_list.append(f"{wikidata_id}: {alias_string}")
    alias_string = ", ".join(alias_string_list)
    return alias_string


def format_sft_sample(question, alias_string, graph_preds, graph_namespaces):
    sft_prompt = f"""
                QUESTION: {question}
                QUESTION ENTITIES: {alias_string}
                KNOWLEDGE GRAPH NAMESPACES: {graph_namespaces}
                KNOWLEDGE GRAPH PREDICATES: {graph_preds}
                """
    return sft_prompt


if __name__ == "__main__":

    rubq_train = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/train_with_aliases.json', 'r'))
    rubq_test = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/test_with_aliases.json', 'r'))

    language = 'en'

    predicate_description_dict = json.load(open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_predicate_mapping.json", 'r'))
    knowledge_graph_predicates = ", ".join(list(predicate_description_dict.keys()))
    knowledge_graph_namespaces = ", ".join(['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'])

    phase = 'train'

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B",
                                              model_max_length=256, use_fast=False)
    sft_examples = []
    failed_samples = []
    for sample in tqdm(rubq_test, total=len(rubq_test)):
        
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
            alias_string = format_id2alias(id2alias, "en")

            user_task = format_sft_sample(question, alias_string, knowledge_graph_predicates, knowledge_graph_namespaces)
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
                
        
            sft_examples.append((sample_id, formatted_prompt))
        else:
            failed_samples.append(sample)
            
    write_tsv(sft_examples, os.path.join("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/rubq_sft_valid.tsv"), expected_num_columns=2)

    print('Total samples: ', len(rubq_train))
    print('Prepared SFT samples: ', len(sft_examples))
    print('Total of failed samples: ', len(failed_samples))








