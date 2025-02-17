import sys
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L12-v2")
es = Elasticsearch(os.getenv('ELASTIC_HOST'))
es = Elasticsearch(
    os.getenv('ELASTIC_HOST'),
    basic_auth=('elastic', os.getenv('ELASTIC_PWD')),
    verify_certs=True,
    ca_certs=os.getenv('ELASTIC_CRT'),
)
index_name = os.getenv('ELASTIC_INDEX'),


def search_full_text(query, top_k=3):
    response = es.search(index=index_name, body={
        "query": {"match": {"content": query}},
        "size": top_k,
    })
    return [
        {
            "document": hit['_source']['document'],
            "page": hit['_source']['page'],
            "content": hit['_source']['content'],
            "score": hit['_score'],
        } 
        for hit in response["hits"]["hits"]
    ]
    

def search_hybride(query_text, top_k=4):
    query_vector = model.encode(query_text).tolist()
    
    # Full-text search query
    full_text_query = {
        "match": {
            "content": query_text
        }
    }

    # Semantic search query
    semantic_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    # Combine both queries in a "should" clause for hybrid search
    hybrid_query = {
        "query": {
            "bool": {
                "should": [
                    full_text_query,
                    semantic_query,
                ]
            }
        },
        "size": top_k,
    }

    response = es.search(index=index_name, body=hybrid_query)

    if len(response) == 0:
        return []

    return [
        {
            "document": hit['_source']['document'],
            "page": hit['_source']['page'],
            "content": hit['_source']['content'],
            "score": hit['_score'],
        } 
        for hit in response["hits"]["hits"]
    ]


def main():
    if len(sys.argv) < 2:
        print("Add a query")
        sys.exit(1)

    search = search_hybride(sys.argv[1])
    for s in search:
        print(f"score : {s['score']} - document : {s['document']} - page : {s['page']}")


if __name__ == "__main__":
    main()
