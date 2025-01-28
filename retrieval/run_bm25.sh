json_resource='resources/global_entities_and_rubq_aliases_dump'
bm25_index='indexes/rubq_bm25_index_with_global'

/home/somov/.conda/envs/bm25_env/bin/python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $json_resource \
  --index $bm25_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 10 \
  --storePositions --storeDocvectors --storeRaw