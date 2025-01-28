import re
import json
import random
from tqdm import tqdm
from prompts import INSTRUCTIONS
from transformers import AutoTokenizer
from preprocessing_utils import preprocess_sparql

def map_wikidata_urls_to_prefix(sparql_query):
    prefix_pattern = r"PREFIX\s+(\w+):\s+<([^>]+)>"
    prefixes = dict(re.findall(prefix_pattern, sparql_query))

    known_prefixes = {
        "http://www.wikidata.org/entity/": "wd:",
        "http://www.wikidata.org/prop/direct/": "wdt:"
    }
    prefix_map = {**known_prefixes, **prefixes}

    def replace_url(match):
        full_url = match.group(1)
        for base_url, prefix in prefix_map.items():
            if full_url.startswith(base_url):
                return prefix + full_url[len(base_url):]
        raise ValueError(f"Unknown URL structure: {full_url}")

    sparql_query = re.sub(prefix_pattern, "", sparql_query)
    url_pattern = r"<(http://www\.wikidata\.org/(entity|prop/direct)/[A-Za-z0-9]+)>"
    return re.sub(url_pattern, replace_url, sparql_query).strip()


# Aliases
def get_alias(entities):
    return {}

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

def preprocess_qald(data, tokenizer, phase='train', lang='en'):
    sft_dataset = []
    failed_samples = []
    all_entities = set()
    all_relations = set()

    for item in data:
        query = item.get('query').get('sparql')
        if query:
            query = preprocess_sparql(map_wikidata_urls_to_prefix(query))
            query_entites = set(re.findall(r'(Q\d+)', query))
            query_relations = set(re.findall(r'(P\d+)', query))
            all_entities.update(query_entites)
            all_relations.update(query_relations)

    for item in tqdm(data):
        query = item.get('query').get('sparql')
        query_entites = set(re.findall(r'(Q\d+)', query))
        id2alias = get_alias(query_entites)

        # Skip question w.o sparql
        if query and (id2alias or True):
            knowledge_graph_predicates = ", ".join(list(all_relations))
            knowledge_graph_namespaces = ", ".join(['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'])

            # clean query from prefixes and urls
            sparql = preprocess_sparql(map_wikidata_urls_to_prefix(query))
            sample_id = str(item['id'])

            question = next(filter(lambda q: q['language'] == lang, item['question']))['string']
            instruction = INSTRUCTIONS[lang]

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

            sft_dataset.append((sample_id, formatted_prompt))
        else:
            failed_samples.append(item)

    with open(f"../data/sft/qald_sft_{phase}.json", 'w') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=4)

    print('Total samples: ', len(data))
    print('Prepared SFT samples: ', len(sft_dataset))
    print('Total of failed samples: ', len(failed_samples))


def preprocess_lcquad(data, tokenizer, phase='train', lang='en'):
    sft_dataset = []
    failed_samples = []
    all_entities = set()
    all_relations = set()

    for item in data:
        query = item.get('query')
        if query:
            query = preprocess_sparql(query)
            query_entites = set(re.findall(r'(Q\d+)', query))
            query_relations = set(re.findall(r'(P\d+)', query))
            all_entities.update(query_entites)
            all_relations.update(query_relations)

    for i, item in enumerate(tqdm(data)):
        query = item.get('query')
        query_entites = set(re.findall(r'(Q\d+)', query))
        id2alias = get_alias(query_entites)

        # Skip question w.o sparql
        if query and (id2alias or True):
            knowledge_graph_predicates = ", ".join(list(all_relations))
            knowledge_graph_namespaces = ", ".join(['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'])

            # clean query from prefixes and urls
            sparql = preprocess_sparql(map_wikidata_urls_to_prefix(query))
            sample_id = str(i)

            question = item.get('en_question')
            instruction = INSTRUCTIONS[lang]

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

            sft_dataset.append((sample_id, formatted_prompt))
        else:
            failed_samples.append(item)

    with open(f"../data/sft/lcquad_sft_{phase}.json", 'w') as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=4)

    print('Total samples: ', len(data))
    print('Prepared SFT samples: ', len(sft_dataset))
    print('Total of failed samples: ', len(failed_samples))

def preprocess_pat(pat_singlehop, pat_multihop, tokenizer, phase='train', lang='en'):
    for q, sample in pat_multihop.items():
        sample['question_type'] = 'multihop'
    for q, sample in pat_singlehop.items():
        sample['question_type'] = 'singlehop'

    full_dict = dict()
    full_dict.update(pat_singlehop)
    full_dict.update(pat_multihop)
    pat_dataset = list(full_dict.values())

    random.shuffle(pat_dataset)
    train_size = int(0.8 * len(pat_dataset))
    pat_train = pat_dataset[:train_size]
    pat_test = pat_dataset[train_size:]

    sft_dataset = []
    failed_samples = []
    all_entities = set()
    all_relations = set()

    for item in pat_dataset:
        query = item.get('query')
        if query:
            query = preprocess_sparql(query)
            # query_entites = set(re.findall(r'(Q\d+)', query))
            question_entities = [item['subject']['subject']]
            # query_relations = set(re.findall(r'(P\d+)', query))
            question_relations = item['relations']
            all_entities.update(question_entities)
            all_relations.update(question_relations)

    for i, item in enumerate(tqdm(pat_dataset)):
        if i >= train_size:
            phase = 'train'
        else:
            phase = 'test'

        query = item.get('query')
        # query_entites = set(re.findall(r'(Q\d+)', query))
        question_entities = [item['subject']['subject']]
        id2alias = get_alias(question_entities)

        # Skip question w.o sparql
        if query and (id2alias or True):
            knowledge_graph_predicates = ", ".join(list(all_relations))
            knowledge_graph_namespaces = ", ".join(['wdt:', 'skos:', 'wd:', 'ps:', 'pq:'])

            # clean query from prefixes and urls
            sparql = preprocess_sparql(query)
            sample_id = str(item['uniq_id'])

            question = item['question']
            instruction = INSTRUCTIONS[lang]

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

            sft_dataset.append((sample_id, formatted_prompt))
        else:
            failed_samples.append(item)

    sft_train = sft_dataset[:train_size]
    sft_test = sft_dataset[train_size:]

    with open(f"../data/sft/pat_sft_train.json", 'w') as f:
        json.dump(sft_train, f, ensure_ascii=False, indent=4)

    with open(f"../data/sft/pat_sft_test.json", 'w') as f:
        json.dump(sft_test, f, ensure_ascii=False, indent=4)

    print('Total samples: ', len(pat_dataset))
    print('Prepared SFT samples: ', len(sft_dataset))
    print('Total of failed samples: ', len(failed_samples))


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", model_max_length=256, use_fast=False)

    # qald_data = json.load(open('../data/qald/qald_train.json'))
    # preprocess_qald(qald_data['questions'], tokenizer, phase='train')

    # pat_singlehop = json.load(open('../data/pat/PAT-singlehop.json'))
    # pat_multihop = json.load(open('../data/pat/PAT-multihop.json'))
    # preprocess_pat(pat_singlehop, pat_multihop, tokenizer)
    #
    # lcquad_data = json.load(open('../data/lcquad/lcquad_2_test.json'))
    # preprocess_lcquad(lcquad_data, tokenizer, phase='test')