import sys
import os
import re
import fitz
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from math import ceil, floor
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L12-v2')
es = Elasticsearch(
    os.getenv('ELASTIC_HOST'),
    basic_auth=('elastic', os.getenv('ELASTIC_PWD')),
    verify_certs=True,
    ca_certs=os.getenv('ELASTIC_CRT'),
)
index_settings = {
    'mappings': {
        'properties': {
            'content': {'type': 'text'},  # Full-text search
            'vector': {'type': 'dense_vector', 'dims': 384}  # Semantic search
        }
    }
}

class IndexingError(Exception):
    pass


def index_content(index_name, title, page, content, url=''):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_settings)
    embedding = model.encode(content).tolist()
    document = {
        'content': content,
        'vector': embedding,
        'document': title,
        'page': page,
        'url': url,
    }
    
    try:
        response = es.index(index=index_name, document=document)

        if response.get('result') in ['created', 'updated']:
            return response['_id']
        else:
            raise IndexingError('Unexpected result: ' + response.get('result'))
    except Exception as e:
        raise IndexingError(f'Indexing failed: {e}')


def extract_toc(doc):
    toc = doc.get_toc()
    return toc

def extract_text_from_pages(doc, page_numbers):
    text = ""
    for page_num in page_numbers:
        page_text = doc[page_num - 1].get_text("text")
        page_text = re.sub(r'\n\d+\n', '', page_text)
        text += page_text
    return text

class Chapter:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def chapters_from_toc(toc, num_pages, level):
    chapters = []
    start = None

    for title in toc:
        if (title[0] <= level) and start != None:
            chapters.append(Chapter(start, title[2]))
            start = None
        if title[0] == level:
            start = title[2]
    
    if start != None:
        chapters.append(Chapter(start, num_pages))
    
    return chapters


def ingest_file(index_name, file_path, file_name, url='', step=5):
    print(file_path)
    doc = fitz.open(file_path)
    num_pages = len(doc)
    toc = extract_toc(doc)
    
    ids = []

    # if table of content is not empty
    if toc != None and len(toc) > 0:
        chapters = chapters_from_toc(toc, num_pages, 3)

        if len(chapters) == 0 or chapters[0].start > 8:
            chapters = chapters_from_toc(toc, num_pages, 2)

        if len(chapters) == 0 or chapters[0].start > 8:
            chapters = chapters_from_toc(toc, num_pages, 1)
            
        if len(chapters) > 0:
            for chapter in chapters:
                text = extract_text_from_pages(doc, range(chapter.start, chapter.end))
                if len(text) / 4 > 8000:
                    try :
                        id = divide_per_chunck(index_name, file_name, url, doc, chapter.start, chapter.end, ceil(len(text) / 4 / 8000))
                        ids.extend(id)
                    except IndexingError:
                        pass
                else:
                    try :
                        id = index_content(index_name, file_name, chapter.start, text, url)
                        ids.append(id)
                    except IndexingError:
                        pass
    
    if len(ids) == 0:
        for page in range(0, num_pages, step):
            page_end = page + step
            if page_end > num_pages:
                page_end = num_pages
            text = extract_text_from_pages(doc, range(page, page_end))

            if len(text) / 4 > 8000:
                try :
                    id = divide_per_chunck(index_name, file_name, url, doc, page, page_end, ceil(len(text) / 4 / 8000))
                    ids.extend(id)
                except IndexingError:
                    pass
            else:
                try :
                    id = index_content(index_name, file_name, page, text, url)
                    ids.append(id)
                except IndexingError:
                    pass
    
    return ids


def divide_per_chunck(index_name, file_name, url, doc, start, end, nb_block):
    ids = []
    for i in range(0, nb_block):
        page_start = floor(start + (end - start) / nb_block * i)
        page_end = floor(start + (end - start) / nb_block * (i + 1))
        text = extract_text_from_pages(doc, range(page_start, page_end))
        try :
            id = index_content(index_name, file_name, page_start, text, url)
            ids.append(id)
        except IndexingError:
            pass
    return ids


def get_documents(index_name, ids):
    try:
        response = es.mget(index=index_name, body={'ids': ids})

        documents = []
        for doc in response['docs']:
            if doc.get('found') and doc.get('_source') is not None:
                documents.append({
                    'document': doc['_source']['document'],
                    'page': doc['_source']['page'],
                    'content': doc['_source']['content'],
                })
        return documents
    except Exception:
        return None

def delete_documents(index_name, ids):
    try:
        actions = []
        for doc_id in ids:
            actions.append({'delete': {'_index': index_name, '_id': doc_id}})

        response = es.bulk(body=actions)

        if not response['errors']:
            return True
        else:
            return False
    except Exception:
        return False

def main():
    if len(sys.argv) < 2:
        print('Add a valid directory')
        sys.exit(1)
    
    if es.ping():
        print('Connected to Elasticsearch !')
        index_name = 'documents'
        es.options(ignore_status=[404]).indices.delete(index=index_name)
        
        directory_path = sys.argv[1]
        if os.path.isdir(directory_path):
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_extension = file.split('.')[-1]
                    if file_extension == 'pdf':
                        file_path = os.path.join(root, file)
                        ids = ingest_file(index_name, file_path, file)
                        print(ids)
        else:
            print(f"'{directory_path}' is not a valid directory")
            sys.exit(1)
    else:
        print('Elasticsearch connection failed.')


if __name__ == "__main__":
    main()
