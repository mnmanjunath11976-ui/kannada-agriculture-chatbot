"""Optional GPU/large-RAM BGE-M3 worker, NOT meant for free Streamlit Cloud.
Run: pip install -r requirements-worker.txt && uvicorn bge_worker:app --host 0.0.0.0 --port 8080
Put behind HTTPS proxy and set BGE_WORKER_TOKEN (never expose publicly without auth).
"""
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from functools import lru_cache
from typing import List

app=FastAPI(title='Private BGE-M3 embedding worker')
class Request(BaseModel):
    texts:List[str]

@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('BAAI/bge-m3',trust_remote_code=False)

@app.post('/embed')
def embed(req:Request,authorization:str=Header(default='')):
    expected=os.environ.get('BGE_WORKER_TOKEN','')
    if not expected or authorization != 'Bearer '+expected:
        raise HTTPException(401,'Authentication required')
    if not 1<=len(req.texts)<=24 or any(len(x)>8000 for x in req.texts):
        raise HTTPException(400,'Maximum 24 texts, 8000 characters each')
    arr=model().encode(req.texts,batch_size=6,normalize_embeddings=True,show_progress_bar=False)
    return {'model':'BAAI/bge-m3','vectors':arr.tolist()}
