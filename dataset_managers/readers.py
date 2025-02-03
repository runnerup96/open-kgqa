import json
from tqdm import tqdm
import preprocessing_utils
from prompts import INSTRUCTIONS
from transformers import AutoTokenizer
def format_id2alias(id2alias, lang='en'):
    alias_string_list = []
    for wikidata_id, alias_lang_dict in id2alias.items():
        alias_string_list.append(f"{wikidata_id} = {alias_lang_dict.get(lang)}\n")
    alias_string = ", ".join(alias_string_list)
    return alias_string


def format_sft_sample(question, alias_string):
    sft_prompt = f"""
                    QUESTION: {question}
                    QUESTION ENTITIES: {alias_string}
                    """
    return sft_prompt

def format_dataset(dataset, tokenizer, relations_description, phase='train', lang='en'):
    sft_examples_list, failed_samples = [], []
    replacement_dict = {}

    for sample in tqdm(dataset):
        sample_id = str(sample['id'])
        question = sample[f'{lang}_question']
        instruction = INSTRUCTIONS[lang]

        id2alias = sample['entities'].get('question')
        if not id2alias:
            id2alias = sample['entities']['query']

        alias_string = format_id2alias(id2alias)
        user_task = format_sft_sample(question, alias_string)

        sparql = sample['query']
        sparql_masked, replacement = preprocessing_utils.replace_preds(sparql, relations_description)
        replacement_dict[sample_id] = replacement

        if alias_string:
            if phase == 'train':
                chat = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_task},
                    {"role": "assistant", "content": sparql_masked}
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

    return sft_examples_list, failed_samples, replacement_dict


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", use_fast=True)
    relations_description = json.load(open("../clean_data/wikidata/relations.json", 'r'))

    dataset = 'qald'

    train_data = json.load(open(f"../clean_data/preprocessed/{dataset}/{dataset}_train.json"))
    test_data = json.load(open(f"../clean_data/preprocessed/{dataset}/{dataset}_test.json"))

    train_sft_examples_list, train_failed_samples, train_replacement_dict = format_dataset(train_data['dataset'], tokenizer, relations_description)
    test_sft_examples_list, valid_failed_samples, test_replacement_dict = format_dataset(test_data['dataset'], tokenizer, relations_description, phase='test')

    json.dump(train_sft_examples_list,
              open(f"../clean_data/sft/{dataset}_train_no_preds.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(train_replacement_dict,
              open(f"../clean_data/sft/{dataset}_train_replacement_dicts.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(test_sft_examples_list,
              open(f"../clean_data/sft/{dataset}_test_no_preds.json", 'w'),
              ensure_ascii=False, indent=4)

    json.dump(test_replacement_dict,
              open(f"../clean_data/sft/{dataset}_test_replacement_dict.json", 'w'),
              ensure_ascii=False, indent=4)


    print('Prepared SFT train samples: ', len(train_sft_examples_list))
    print('Total of train failed samples: ', len(train_failed_samples))

    print('Prepared SFT train samples: ', len(test_sft_examples_list))
    print('Total of test failed samples: ', len(test_sft_examples_list))






