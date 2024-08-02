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
