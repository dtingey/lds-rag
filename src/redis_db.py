import os
import json
import time

import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer


# url = "https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started/main/data/bikes.json"
# response = requests.get(url)
# bikes = response.json()

data_dir = "/Users/damontingey/personal/lds-rag/data/scriptures/flat"

data = []
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    with open(file_path) as f:
        data.append(json.load(f))

verses = []
for book in data:
    verses += book["verses"]

json.dumps(verses, indent=2)

client = redis.Redis(
  host='redis-13320.c309.us-east-2-1.ec2.redns.redis-cloud.com',
  port=13320,
  password=os.environ['REDIS_CLOUD_PASSWORD'],
  decode_responses=True)

res = client.ping()
# >>> True

# clean db
client.flushall()

pipeline = client.pipeline()
for i, verse in enumerate(verses, start=1):
    redis_key = f"verses:{i:05}"
    pipeline.json().set(redis_key, "$", verse)
res = pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

breakpoint()

res = client.json().get("verses:00010", "$.reference")
# >>> ['Summit']

keys = sorted(client.keys("verses:*"))
# >>> ['bikes:001', 'bikes:002', ..., 'bikes:011']

texts = client.json().mget(keys, "$.text")
breakpoint()

texts = [item for sublist in texts for item in sublist]
embedder = SentenceTransformer("msmarco-distilbert-base-v4")
embeddings = embedder.encode(texts).astype(np.float32).tolist()
VECTOR_DIMENSION = len(embeddings[0])
# >>> 768

pipeline = client.pipeline()
for key, embedding in zip(keys, embeddings):
    pipeline.json().set(key, "$.text_embedding", embedding)
pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

res = client.json().get("verses:11111")
# >>>
# {
#   "reference": "1 Nephi 1:1",
#   "text": "I Nephi, having been born of goodly parents..."
#   "text_embedding": [
#     -0.538114607334137,
#     -0.49465855956077576,
#     -0.025176964700222015,
#     ...
#   ]
# }

schema = (
    TextField("$.reference", no_stem=True, as_name="reference"),
    TextField("$.text", as_name="text"),
    VectorField(
        "$.text_embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
definition = IndexDefinition(prefix=["verses:"], index_type=IndexType.JSON)
res = client.ft("idx:verses_vss").create_index(
    fields=schema, definition=definition
)
# >>> 'OK'

info = client.ft("idx:verses_vss").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]
# print(f"{num_docs} documents indexed with {indexing_failures} failures")
# >>> 11 documents indexed with 0 failures

breakpoint()

query = Query("@:Peaknetic")
res = client.ft("idx:bikes_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:008', 'payload': None, 'brand': 'Peaknetic', 'model': 'Soothe Electric bike', 'price': '1950', 'description_embeddings': ...

query = Query("@brand:Peaknetic").return_fields("reference", "text")
res = client.ft("idx:verses_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:008', 'payload': None, 'brand': 'Peaknetic', 'model': 'Soothe Electric bike', 'price': '1950'}, Document {'id': 'bikes:009', 'payload': None, 'brand': 'Peaknetic', 'model': 'Secto', 'price': '430'}]

query = Query("@brand:Peaknetic @price:[0 1000]").return_fields(
    "id", "brand", "model", "price"
)
res = client.ft("idx:bikes_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:009', 'payload': None, 'brand': 'Peaknetic', 'model': 'Secto', 'price': '430'}]

queries = [
    "Bike for small kids",
    "Best Mountain bikes for kids",
    "Cheap Mountain bike for kids",
    "Female specific mountain bike",
    "Road bike for beginners",
    "Commuter bike for people over 60",
    "Comfortable commuter bike",
    "Good bike for college students",
    "Mountain bike for beginners",
    "Vintage bike",
    "Comfortable city bike",
]

encoded_queries = embedder.encode(queries)
len(encoded_queries)
# >>> 11


def create_query_table(query, queries, encoded_queries, extra_params={}):
    results_list = []
    for i, encoded_query in enumerate(encoded_queries):
        result_docs = (
            client.ft("idx:bikes_vss")
            .search(
                query,
                {
                    "query_vector": np.array(
                        encoded_query, dtype=np.float32
                    ).tobytes()
                }
                | extra_params,
            )
            .docs
        )
        for doc in result_docs:
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": queries[i],
                    "score": vector_score,
                    "id": doc.id,
                    "brand": doc.brand,
                    "model": doc.model,
                    "description": doc.description,
                }
            )

    # Optional: convert the table to Markdown using Pandas
    queries_table = pd.DataFrame(results_list)
    queries_table.sort_values(
        by=["query", "score"], ascending=[True, False], inplace=True
    )
    queries_table["query"] = queries_table.groupby("query")["query"].transform(
        lambda x: [x.iloc[0]] + [""] * (len(x) - 1)
    )
    queries_table["description"] = queries_table["description"].apply(
        lambda x: (x[:497] + "...") if len(x) > 500 else x
    )
    queries_table.to_markdown(index=False)



query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

create_query_table(query, queries, encoded_queries)
# >>> | Best Mountain bikes for kids     |    0.54 | bikes:003... (+ 32 more results)

hybrid_query = (
    Query("(@brand:Peaknetic)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)
create_query_table(hybrid_query, queries, encoded_queries)
# >>> | Best Mountain bikes for kids     |    0.3  | bikes:008... (+22 more results)

range_query = (
    Query(
        "@vector:[VECTOR_RANGE $range $query_vector]=>{$YIELD_DISTANCE_AS: vector_score}"
    )
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .paging(0, 4)
    .dialect(2)
)
create_query_table(
    range_query, queries[:1], encoded_queries[:1], {"range": 0.55}
)
# >>> | Bike for small kids |    0.52 | bikes:001 | Velorim    |... (+1 more result)
