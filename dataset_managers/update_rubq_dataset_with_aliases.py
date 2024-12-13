
import plyvel
import json
import requests
import copy
import pickle

import tqdm





def get_alias_from_wikidata(entity_id, lang):
    """
    Fetches English aliases for a given Wikidata entity ID.

    Parameters:
        entity_id (str): The Wikidata entity ID (e.g., 'Q42' for Douglas Adams).

    Returns:
        list: A list of aliases in English.
    """
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "aliases",
        "format": "json",
        "languages": lang  # Restrict to English
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract English aliases
        aliases = []
        if "entities" in data and entity_id in data["entities"]:
            entity_data = data["entities"][entity_id]
            if "aliases" in entity_data and lang in entity_data["aliases"]:
                aliases = [alias["value"] for alias in entity_data["aliases"][lang]]

        return aliases
    except requests.RequestException as e:
        print(f"Error fetching data from Wikidata: {e}")
        return []

if __name__ == "__main__":
    lang = 'en'

    rubq_train = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/train.json'))
    rubq_test = json.load(open('/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/test.json'))

    wikidata_level_db = plyvel.DB('/Users/somov-od/Downloads/aliases_lvldb')

    updated_test_with_aliases = []
    samples_not_in_leveldb = []
    samples_with_no_qentities = []
    for sample in tqdm.tqdm(rubq_test, total=len(rubq_test)):
        new_sample = copy.deepcopy(sample)
        if new_sample['question_uris'] and len(new_sample['question_uris']) > 0:
            entity_id_list = [sample.split('/')[-1] for sample in new_sample['question_uris']]

            id2alias = dict()
            for wikidata_id in entity_id_list:
                aliases = wikidata_level_db.get(wikidata_id.encode())
                if aliases is None:
                    aliases = get_alias_from_wikidata(wikidata_id, 'en')
                    if aliases:
                        id2alias[wikidata_id] = {lang: aliases}
                    # continue
                else:
                    alias_dict = pickle.loads(aliases)
                    alias_dict = {k: list(v) for k,v in alias_dict.items()}
                    id2alias[wikidata_id] = alias_dict

            if len(id2alias) != len(entity_id_list):
                samples_not_in_leveldb.append(new_sample)
            else:
                new_sample['id2alias'] = id2alias
                updated_test_with_aliases.append(new_sample)
        else:
            samples_with_no_qentities.append(sample)

    json.dump(updated_test_with_aliases, open("/Users/somov-od/Documents/phd/projects/open_kgqa/data/RuBQ/RuBQ_2.0/test_with_aliases.json", 'w'),
              ensure_ascii=False, indent=4)

    print('Total of samples: ', len(rubq_test))
    # samples
    print('Entities not in level_db: ', len(samples_not_in_leveldb))
    print('Entities without qids: ', len(samples_with_no_qentities))
    print('Good samples: ', len(updated_test_with_aliases))







