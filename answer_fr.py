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
    prompt = f"Extraire les mots clés de cette question :\n{query}\nVotre réponse sera une liste avec les mots clés séparés par une virgule. Si aucun mot clé n'est trouvé, renvoyer 0."
    response = client.chat.completions.create(
        model=os.getenv('LLM_MODEL'),
        messages=[
            {"role": "system", "content": """
Vous êtes un assistant pour extraire des mots clés.
Si des mots sont en majuscules dans la question, c'est qu'il s'agit de mots clés.
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
        doc_context = f"Titre du document : {s['document']}, Page : {s['page']}, Texte : {s['content']}\n"
        if (len(context) + len(doc_context)) / 4 < int(os.getenv('LLM_NB_TOKENS')):
            context += doc_context

    prompt = f"A partir des documents, répondre à la question : {question}"
    response = client.chat.completions.create(
        model=os.getenv('LLM_MODEL'),
        messages=[
            {"role": "system", "content": f"""
Vous êtes un assistant qui répond aux qestions en vous basant uniquement sur les documents suivants.
Vous citerez toujours vos sources à la fin de la réponse en précisant le titre exacte des documents et la page pour chaque document.
Si les documents ne vous permettent pas de répondre, ne pas inventer d'information et répondre que vous n'êtes pas en mesure de répondre.
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
