from pprint import pprint
import os
from elasticsearch import Elasticsearch, helpers
import json
data = []
count = 0
for root, dirs, files in os.walk("./lyrics", topdown=False):
    for name in files:
        count += 1
        path = (os.path.join(root, name))
        data.append({
            "_index": "animesonglyrics",
            "_id": str(count),
            # "_type": "song",
            "_source": {
                "name": name,
                "text": open(path, 'r',).read()
            }
        })

# with open('a.json', 'w+') as f:
#     f.write(json.dumps(data, ensure_ascii=False))

# data = json.load(open('a.json'))
# data = data[:2]

es = Elasticsearch()


# helpers.bulk(es, data)
# while True:
#     text = input("> ")
#     res = es.search(index="animesonglyrics",
#                     body={"query": {"match": {"text": text}}})
#     for i in res['hits']['hits']:
#         print("title", i['_source']['name'])
# print([i['_source']['text']][0])
