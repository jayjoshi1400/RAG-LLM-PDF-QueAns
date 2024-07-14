from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import sqlite3
from vector_loader import file_processed_check, get_docs, get_chunks, get_vector_store
from query_llm import query_rag

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize database connection
conn = sqlite3.connect('processed_files.db')
c = conn.cursor()

class QueryModel(BaseModel):
    query: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.post("/process/")
async def process_files():
    include_list = file_processed_check(UPLOAD_DIR)
    docs = get_docs(UPLOAD_DIR, include_list)
    chunks = get_chunks(docs)
    get_vector_store(chunks)
    
    return {"info": "Files processed successfully"}

@app.get("/status/")
async def status():
    c.execute("SELECT * FROM processed_files")
    processed_files = c.fetchall()
    return {"processed_files": processed_files}

@app.post("/query/")
async def query_endpoint(query: QueryModel):
    response = query_rag(query.query)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
