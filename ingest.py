import sys
import os
import re
import fitz
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from math import ceil, floor
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L12-v2")
es = Elasticsearch(
    os.getenv('ELASTIC_HOST'),
    basic_auth=('elastic', os.getenv('ELASTIC_PWD')),
    verify_certs=True,
    ca_certs=os.getenv('ELASTIC_CRT'),
)
index_name = os.getenv('ELASTIC_INDEX'),
index_settings = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},  # Full-text search
            "vector": {"type": "dense_vector", "dims": 384}  # Semantic search
        }
    }
}

def index_content(file_path, page, content):
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_settings)
    embedding = model.encode(content).tolist()
    body = {
        "content": content,
        "vector": embedding,
        "document": file_path,
        "page": page,
        "url": "",
    }
    es.index(index=index_name, body=body)

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


def ingest_file(file_path, file_name, step=5):
    print(file_path)
    doc = fitz.open(file_path)
    num_pages = len(doc)
    toc = extract_toc(doc)
    
    with_toc = False

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
                    divide_per_chunck(file_name, doc, chapter.start, chapter.end, ceil(len(text) / 4 / 8000))
                else:
                    index_content(file_name, chapter.start, text)
            with_toc = True
    
    if not with_toc:
        for page in range(0, num_pages, step):
            page_end = page + step
            if page_end > num_pages:
                page_end = num_pages
            text = extract_text_from_pages(doc, range(page, page_end))

            if len(text) / 4 > 8000:
                divide_per_chunck(file_name, doc, page, page_end, ceil(len(text) / 4 / 8000))
            else:
                index_content(file_name, page, text)


def divide_per_chunck(file_name, doc, start, end, nb_block):
    for i in range(0, nb_block):
        page_start = floor(start + (end - start) / nb_block * i)
        page_end = floor(start + (end - start) / nb_block * (i + 1))
        text = extract_text_from_pages(doc, range(page_start, page_end))
        index_content(file_name, page_start, text)


def main():
    if len(sys.argv) < 2:
        print("Add a valid directory")
        sys.exit(1)
    
    if es.ping():
        print('Connected to Elasticsearch !')
        es.options(ignore_status=[404]).indices.delete(index=index_name)
        
        directory_path = sys.argv[1]
        if os.path.isdir(directory_path):
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_extension = file.split('.')[-1]
                    if file_extension == 'pdf':
                        file_path = os.path.join(root, file)
                        ingest_file(file_path, file)
        else:
            print(f"'{directory_path}' is not a valid directory")
            sys.exit(1)
    else:
        print('Elasticsearch connection failed.')


if __name__ == "__main__":
    main()
