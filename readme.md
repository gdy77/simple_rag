# Simple RAG

RAG (Retrieval-Augmented Generation) project that uses Elasticsearch to retrieve context from PDF files and deliver it to an LLM (Large Language Model) for enhanced response generation.
Live demo (in french) : https://search.godefroy.tech/


### Install and run

Install the project :
`pip install -r requirements.txt`

Install elastic search :
https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

Index pdf in elastic search :
`python ingest.py <directory with pdf>`

Search with elastic search :
`python search.py "query"`

Connect your llm

Generate answers with context :
`python answers_en.py "query"`
or `python answers_fr.py "query"`


### Environment variable

ELASTIC_HOST
ELASTIC_PWD
ELASTIC_CRT
LLM_URL
LLM_KEY
LLM_MODEL
LLM_NB_TOKENS
