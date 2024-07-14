import streamlit as st
import sqlite3
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from embeddings_function import get_embeddings


load_dotenv()

CHROMA_PATH = "chroma"

# Initializing db for storing processed files
conn = sqlite3.connect('processed_files.db')
c = conn.cursor()

# Create table to store file details
c.execute('''CREATE TABLE IF NOT EXISTS processed_files
             (filename text)''')


# Initial check to see if files have been already processed.
def file_processed_check(pdf_dir):
    files_to_exclude = []
    # Finding the file names
    all_files = os.listdir(pdf_dir)
    # Filtering out files that are not PDFs
    pdf_files = [file for file in all_files if file.lower().endswith('.pdf')]
    # print(pdf_files)
    for name in pdf_files:
        print(name)
        c.execute('SELECT * FROM processed_files WHERE filename=?', (name,))
        res = c.fetchone()
        if res is not None:
            file = "data\\"+str(name)
            print('Excluding file: ', file) 
            files_to_exclude.append(file)
        else:
            file = "data\\"+str(name)
            print('Adding file: ', file)
    include_files = [f for f in all_files if f not in files_to_exclude]
    return include_files

def get_docs(pdf_folder, include_files):
    # No files need to be included
    if not include_files:
        return None
    # Only inlcude these files
    else:
        filter = ("["+("|".join(include_files))+"]*")
        filter = filter.replace('.pdf', '')
    data = PyPDFDirectoryLoader(pdf_folder, glob=filter)
    loader = data.load()
    return loader

def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_splitter.split_documents(docs)

    

def get_chunk_id(chunks):
    # IDs of the type: page_source:page_number:chunk_id
    temp_pg_id = -1
    cur_chunk = 0

    # metadata example: metadata={'source': 'data\\paper1.pdf', 'page': 10}
    for chunk in chunks:
        src = chunk.metadata.get('source')
        pg = chunk.metadata.get('page')
        cur_pg_id = f"{src}:{pg}"

        if cur_pg_id == temp_pg_id:
            cur_chunk += 1
        else:
            cur_chunk = 0
        
        cur_id = f"{src}:{pg}:{cur_chunk}"
        temp_pg_id = cur_pg_id

        chunk.metadata['id'] = cur_id

    return chunks



def add_processed_file(filename):
    with sqlite3.connect('processed_files.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO processed_files (filename) VALUES (?)", (filename,))
        conn.commit()

'''
Functionalities:
1. Map each chunk with unique IDs based on source and page number (and further id if on the same page)
2. If new chunks are found, add to the db. If chunks are same, skip them.
'''
def get_vector_store(chunks):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    chunk_id = get_chunk_id(chunks)
    old_chunks = db.get(include=[])
    old_id = set(old_chunks['ids'])
    new_chunks = []
    processed_srcs = set()

    for chunk in chunk_id:
        src = chunk.metadata['source']
        if chunk.metadata['id'] not in old_id:
            new_chunks.append(chunk)
            processed_srcs.add(os.path.basename(src))
    
    # print(processed_srcs)
    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_id = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_id)
    else:
        print('no new docs')
    for i in processed_srcs:
        add_processed_file(i)
        print('Added file to SQL DB: ', i)



def main():
    include_list = file_processed_check('data')
    docs = get_docs('data', include_list)
    chunks = get_chunks(docs)
    get_vector_store(chunks)
    

if __name__ == "__main__":
    main()