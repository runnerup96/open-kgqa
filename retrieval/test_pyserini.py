
import json
import time
from pyserini.search.lucene import LuceneSearcher


if __name__ == "__main__":


    d = []

    json.dump(d, open('index_dumps/entities_index/just_labels_and_rubq_aliases_dump.json', 'w'),
              ensure_ascii=False, indent=4)

    index_dir = "/home/somov/open_kgqa/retrieval/index_dumps/entities_index"

    print('Begin building Lucene Index...')
    time_start = time.time()
    searcher = LuceneSearcher(index_dir)
    time_end = time.time()
    print(f'Index is built in {round(time_end - time_start, 3)}')

    k1, b = 0.4, 0.4
    searcher.set_bm25(k1, b)

    searcher = LuceneSearcher('indexes/sample_collection_jsonl')
    hits = searcher.search('document')

    for i in range(len(hits)):
        print(f'{i + 1:2} {hits[i].docid:4} {hits[i].score:.5f}')