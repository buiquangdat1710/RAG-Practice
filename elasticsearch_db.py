# elasticsearch_db.py
from elasticsearch import Elasticsearch

class ElasticSearchDB:
    def __init__(self, host="http://localhost:9200", index_name="products"):
        self.es = Elasticsearch(hosts=[host])
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name)

    def insert_document(self, document):
        self.es.index(index=self.index_name, document=document)

    def count_documents(self):
        return self.es.count(index=self.index_name)['count']

    def search(self, query, limit=5):
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "information"]
                }
            }
        }
        response = self.es.search(index=self.index_name, body=body, size=limit)
        results = [
            {
                "title": hit["_source"]["title"],
                "information": hit["_source"]["information"]
            }
            for hit in response["hits"]["hits"]
        ]
        return results
