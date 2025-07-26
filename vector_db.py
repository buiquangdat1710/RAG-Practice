
from pymongo import MongoClient
from chromadb import HttpClient
from qdrant_client import QdrantClient
from supabase import create_client, Client
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models
load_dotenv()
import os

load_dotenv()

class VectorDatabase:
    def __init__(self, db_type: str):
        self.db_type = db_type
        if self.db_type == "mongodb":
            self.client = MongoClient(os.getenv("MONGODB_URI"))
        elif self.db_type == "chromadb":
            self.client = HttpClient(
                host="localhost", 
                port=8123
            )
        elif self.db_type == "qdrant":
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_KEY"),
            )
        elif self.db_type == "supabase":
            url: str = os.environ.get("SUPABASE_URL")
            key: str = os.environ.get("SUPABASE_KEY")
            supabase: Client = create_client(
                supabase_url=url,
                supabase_key=key
                )
            self.client = supabase
    def insert_document(self, collection_name: str, document: dict):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            collection.insert_one(document)
        elif self.db_type == "chromadb":
            self.client.add_documents(
                collection_name=collection_name,
                documents=[document]
            )
        elif self.db_type == "qdrant":
            self.client.upsert(
                collection_name=collection_name,
                points=[qdrant_models.PointStruct(id=document['title'], vector=document['embedding'], payload=document)]
            )
        elif self.db_type == "supabase":
            self.client.table(collection_name).insert(document).execute()
    def query(self, collection_name: str, query_vector: list, limit: int = 5):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # tên index bạn đã tạo
                        "queryVector": query_vector,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": limit
                    }
                }
            ])
            return list(results)
        elif self.db_type == "chromadb":
            results = self.client.query(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results
        elif self.db_type == "qdrant":
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").execute()
            return response.data
    def document_exists(self, collection_name, filter_query):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return collection.count_documents(filter_query) > 0
        elif self.db_type == "chromadb":
            # ChromaDB không hỗ trợ document-level metadata query, return False để luôn insert
            return False
        elif self.db_type == "qdrant":
            result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={"must": [{"key": k, "match": {"value": v}} for k, v in filter_query.items()]},
                limit=1
            )
            return len(result[0]) > 0
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").eq("title", filter_query["title"]).execute()
            return len(response.data) > 0
        else:
            raise ValueError("Unsupported database type")
