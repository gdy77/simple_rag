import os
import sys
from openai import OpenAI
from search import search_hybride
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url = os.getenv('LLM_URL'),
    api_key = os.getenv('LLM_KEY'),
)

def answer_keywords(query):
    prompt = f"Extract the keywords from this question. :\n{query}\nYour answer will be a list with the keywords separated by a comma. If no keyword is found, return 0."
    response = client.chat.completions.create(
        model=os.getenv('LLM_MODEL'),
        messages=[
            {"role": "system", "content": """
You are an assistant for extracting keywords.
If words are in uppercase in the question, they are keywords.
            """},
            {"role": "user", "content": prompt}
        ]
    )
    keywords = response.choices[0].message.content.split(",")
    
    if len(keywords) == 0 or (len(keywords) == 1 and '0' in keywords[0]):
        return []

    return [k.strip() for k in keywords]


def answer_generate(question, search_results):
    context = ""
    for s in search_results:
        doc_context = f"Document title : {s['document']}, Page : {s['page']}, Text : {s['content']}\n"
        if (len(context) + len(doc_context)) / 4 < int(os.getenv('LLM_NB_TOKENS')):
            context += doc_context

    prompt = f"From the documents, answer the question : {question}"
    response = client.chat.completions.create(
        model=os.getenv('LLM_MODEL'),
        messages=[
            {"role": "system", "content": f"""
You are an assistant who answers questions based solely on the following documents.
You will always cite your sources at the end of the answer, specifying the exact title of the documents and the page number for each document.
If the documents do not allow you to answer, do not make up any information and respond that you are not able to answer.
\n
{context}
"""
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("Add a query")
        sys.exit(1)

    query = sys.argv[1]
    keywords = answer_keywords(query)

    if len(keywords) > 0:
        print(" ".join(keywords))
        search = search_hybride(" ".join(keywords))
        #search = search_hybride(query)
        for s in search:
            print(f"score : {s['score']} - document : {s['document']} - page : {s['page']}")

        if len(search) > 0:
            answer = answer_generate(query, search)
            print(answer)
        else:
            print("Can not find any information about this question")
    else:
        print("Fail to find keywords")
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
