import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from embeddings_function import get_embeddings
import google.generativeai as genai


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Contextual Information:

{context}

Based on the contextual information provided above, answer the following question: {question} 
"""

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

def query_rag(query):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    # Performs similarity search on the vector db to return the top 3 most similar documents
    res = db.similarity_search_with_score(query, k=3)

    context = "\n---\n".join([doc.page_content for doc, _score in res])
    prompt_temp = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_temp.format(context=context, question=query)
    print('\n\n Prompt: \n')
    print(prompt)
    print('\n\n')

    response = model.generate_content(prompt)

    srcs = [doc.metadata.get("id", None) for doc, _score in res]
    formatted_response = {
        "response": response.text,
        "sources": srcs
    }
    return formatted_response