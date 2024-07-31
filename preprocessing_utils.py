import numpy as np
import re

def is_schema_token(token, schema_componenents):
    for ns in schema_componenents:
        if token.startswith(ns):
            return True
    return False


def preprocess_sparql(sparql):
    # make tokenization easier
    sparql = sparql.replace('\n', ' ')
    sparql = sparql.replace('{', ' { ')
    sparql = sparql.replace('}', ' } ')
    sparql = sparql.replace('(', ' ( ')
    sparql = sparql.replace(')', ' ) ')
    sparql = sparql.replace('[', ' [ ')
    sparql = sparql.replace(']', ' ] ')
    sparql = sparql.replace(',', ' , ')
    sparql = sparql.replace('.', ' . ')
    sparql = sparql.replace('|', ' | ')
    sparql = sparql.replace('/', ' / ')
    sparql = sparql.replace(';', ' ; ')


    # sparql = sparql.replace(' ?', '_?')

    sparql = sparql.strip()
    sparql_tokens = sparql.split()
    updated_lower_sparql = []
    for token in sparql_tokens:
        token = token.strip()
        if is_schema_token(token, ['dr:', 'wd:', 'wdt:', 'p:', 'pq:', 'ps:', 'psn:']) == False:
            updated_lower_sparql.append(token.lower())
        else:
            updated_lower_sparql.append(token)

    updated_lower_sparql = " ".join(updated_lower_sparql).strip()
    updated_lower_sparql = updated_lower_sparql.replace('. }', ' }')

    return updated_lower_sparql


def find_query_attrs(sparql):
    attr_tokens = []
    sparql_tokens = sparql.split(" ")
    i = 0
    while sparql_tokens[i] != '{':
        if sparql_tokens[i].startswith('?'):
            attr_tokens.append(sparql_tokens[i])
        i += 1
    return attr_tokens


def map_to_meaning_full_attrs(sparql):
    # определяем что является аттрибутом
    # определяем триплеты и их компоненты
    attr_tokens = find_query_attrs(sparql)

    sparql_tokens = sparql.split()

    start_index = sparql_tokens.index('{')

    attr_mapping = dict()

    pred_indicator = ['wdt:', 'p:', 'pq:', 'ps:']
    specific_sparql_tokens = sparql_tokens[start_index:]
    # определяем неизвестные сущности и
    for i in range(len(specific_sparql_tokens)):
        if is_schema_token(specific_sparql_tokens[i], ['dr:', 'wdt:', 'p:', 'pq:', 'ps:', 'rdfs:']):
            subj, pred, obj = specific_sparql_tokens[i - 1], specific_sparql_tokens[i], specific_sparql_tokens[i + 1]
            if subj == '(' and obj == ')':
                subj, pred, obj = specific_sparql_tokens[i - 2], specific_sparql_tokens[i], specific_sparql_tokens[
                    i + 2]

            # нужно знание текущего места и индекса
            if subj.startswith('?') or subj.startswith('wd:Q'):
                if subj not in attr_mapping:
                    attr_mapping[subj] = {
                        "triplet_place": ['SUBJECT'],
                        "attr_idx": len(attr_mapping) + 1,
                        "curr_iter": 0
                    }
                else:
                    attr_mapping[subj]['triplet_place'].append('SUBJECT')

            if obj.startswith('?') or obj.startswith('wd:Q'):
                if obj not in attr_mapping:
                    attr_mapping[obj] = {
                        "triplet_place": ['OBJECT'],
                        "attr_idx": len(attr_mapping) + 1,
                        "curr_iter": 0
                    }
                else:
                    attr_mapping[obj]['triplet_place'].append('OBJECT')

    # print(unknown_attr_mapping)

    result_sparql_tokens = []
    for token in sparql_tokens:
        if token in attr_mapping:
            if token in attr_tokens:
                token_place, token_idx = attr_mapping[token]['triplet_place'][-1], attr_mapping[token]['attr_idx']
                attr_tokens.remove(token)
            else:
                token_list, token_idx, curr_iter = attr_mapping[token]['triplet_place'], \
                                                   attr_mapping[token]['attr_idx'], \
                                                   attr_mapping[token]['curr_iter']

                # обработка для filter/order - все что после триплетов
                if curr_iter < len(attr_mapping[token]['triplet_place']):
                    token_place = token_list[curr_iter]
                else:
                    token_place = token_list[-1]
                attr_mapping[token]['curr_iter'] += 1

            if token.startswith('?'):
                formated_attr = f'?{token_place}_{token_idx}'
            elif token.startswith('wd:Q'):
                formated_attr = f'{token_place}_{token_idx}'
            result_sparql_tokens.append(formated_attr)
        else:
            result_sparql_tokens.append(token)
    # TODO: ключ curr_iter - удалить из attr_mapping словаря
    final_sparql = " ".join(result_sparql_tokens).strip()
    final_sparql = final_sparql.replace(' ?', '_?')
    return final_sparql, attr_mapping

def simple_entity_mapping(sparql):
    entities = re.findall(r"wd:Q\d+", sparql)
    entity_map = dict()
    used_entities_set = set()
    for idx, entity in enumerate(entities):
        idx += 1
        if entity not in used_entities_set:
            sparql = sparql.replace(entity, f'ENTITY_{idx}')
            entity_map[f'ENTITY_{idx}'] = entity
            used_entities_set.add(entity)
    return sparql, entity_map

def form_masked_query(sparql):
    pr_sparql = preprocess_sparql(sparql)
    mapped_sparql, attr_mapping = simple_entity_mapping(pr_sparql)
    return {"masked_query": mapped_sparql, "attr_mapping": attr_mapping}


def apply_augmentation(text, word_aug, char_aug):
    aug_number = np.random.randint(4)
    if aug_number == 0:
        aug_text = word_aug.augment(text=text, action="stopword")
    elif aug_number == 1:
        aug_text = char_aug.augment(text=text, action="orfo")
    elif aug_number == 2:
        aug_text = char_aug.augment(text=text, action="typo")
    elif aug_number == 3:
        aug_text = char_aug.augment(text=text, action="multiply")

    return aug_text