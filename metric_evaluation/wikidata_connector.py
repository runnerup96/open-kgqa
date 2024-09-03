import json
from time import sleep
import pandas as pd
from tqdm import tqdm
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def replace(x):
    if x:
        x = x.replace("http://www.wikidata.org/entity/statement/", 'wds:')
        x = x.replace("http://www.wikidata.org/entity/", 'wd:')
        x = x.replace("http://www.w3.org/2000/01/rdf-schema#", 'rdfs:')
        x = x.replace("http://tkles-pcb000239.vm.esrt.cloud.sbrf.ru:8082/object/", 'dr:')
        x = x.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", 'rdfs:')
        x = x.replace("http://deepreason.ai/kb/chitchat/", "dr:")
    return x



class BlazegraphConnector:
    def __init__(self, url="http://query.wikidata.org/sparql"):
        self.url = url

    def delete_bnode(self, rec):
        if rec.get('type', None) == 'bnode':
            rec['value'] = None
        return rec

    @staticmethod
    def send_get_request(url, query):
        sess = requests.session()
        retries = Retry(total=2,
                        backoff_factor=3,
                        status_forcelist=[429, 500, 502, 503, 504])
        sess.mount('https://', HTTPAdapter(max_retries=retries))
        res = sess.get(
            url = url,
            params={'query': query},
            headers={'Accept': 'application/sparql-results+json'}
        ).json()
        return res

    def fetch_query(self, query):
        if not query:
            return None
        try:
            sleep(1)
            records = self.send_get_request(self.url, query)
        except Exception as e:
            return "Exception: " + str(e)

        header = sorted(records['head']['vars'])
        records = list(map(lambda rec: {name: self.delete_bnode(rec.get(name, {}))
                                        for name in header},
                           records['results']['bindings']))
        records = list(map(lambda rec: [
            replace(rec.get(name, {}).get('value')) for name in header],
                           records))

        records = pd.DataFrame.from_records(records, columns=header)
        return records.to_numpy().flatten().tolist()




if __name__=='__main__':
    bgc = BlazegraphConnector()
    q = 'SELECT ?answer\nWHERE {\n  wd:Q35637 p:P1346 [ ps:P1346 ?answer; pq:P585 "1906-01-01T00:00:00+00:00"^^xsd:dateTime ]\n}'
    print(bgc.fetch_query(q))

    message = input('SparQL: ')
    while message != ':q':
        print(bgc.fetch_query(message))
        message = input('SparQL: ')