"""Persistent provider-separated neural embeddings for Kannada Agriculture V7.
Never modifies knowledge/v2 document records. No work on import. Storage is GitHubStore.get/put.
Dense vectors use normalized float16 for compact, versioned private storage.
"""
import base64
import hashlib
import json
import re
from pathlib import PurePosixPath
import numpy as np
import requests

ROOT = "knowledge/v3_semantic"
PROVIDERS = {"Gemini Embedding 2": "gemini2", "BGE-M3 endpoint": "bge_m3"}


def provider_label(key):
    return {v:k for k,v in PROVIDERS.items()}.get(key, key)


def chunk_id(row):
    # Exact text+source fingerprint means changes are re-embedded; previous vectors remain isolated.
    payload=json.dumps([row.get('docid',''),row.get('page',''),row.get('source_location',''),row.get('kn',''),row.get('en','')],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]


def embedding_text(row):
    kn=(row.get('kn') or '').strip(); en=(row.get('en') or '').strip()
    return ('Kannada agricultural document:\n'+kn+'\nEnglish translation:\n'+en)[:6500] if en else ('Kannada agricultural document:\n'+kn)[:6500]


def _pack(v):
    a=np.asarray(v,dtype=np.float32)
    if a.ndim!=1 or len(a)<128 or not np.isfinite(a).all():
        raise ValueError('Invalid embedding dimension or values')
    norm=np.linalg.norm(a)
    if norm<1e-10: raise ValueError('Zero embedding vector')
    a=a/norm
    return base64.b64encode(a.astype(np.float16).tobytes()).decode('ascii'),len(a)


def _unpack(s,dim):
    a=np.frombuffer(base64.b64decode(s),dtype=np.float16).astype(np.float32)
    if len(a)!=dim: raise ValueError('Stored vector dimension mismatch')
    norm=np.linalg.norm(a)
    return a/(norm+1e-12)


def _read(store,path,default):
    data,_=store.get(path)
    return json.loads(data.decode('utf-8')) if data else default


def _write(store,path,data,reason):
    raw=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    if len(raw)>800000: raise ValueError('Vector shard exceeds private GitHub file size policy')
    store.put(path,raw,reason)


def _path(provider):
    if provider not in PROVIDERS.values(): raise ValueError('Unknown embedding provider')
    return f'{ROOT}/{provider}'


def load_manifest(store,provider):
    return _read(store,_path(provider)+'/manifest.json',{'version':1,'provider':provider,'dim':None,'shards':[],'count':0})


def load_vectors(store,provider):
    meta=load_manifest(store,provider)
    records={}
    for item in meta.get('shards',[]):
        shard=_read(store,_path(provider)+'/'+item,{'records':[]})
        for row in shard.get('records',[]):records[row['id']]=row
    return meta,records


def embed_gemini(client,texts,model='gemini-embedding-2'):
    # Embedding 2 does NOT accept the older task_type parameter.
    vectors=[]
    for text in texts:
        r=client.models.embed_content(model=model,contents=text)
        if not r.embeddings: raise RuntimeError('Empty Gemini embedding result')
        vectors.append(r.embeddings[0].values)
    return vectors


def embed_bge_endpoint(texts,url,token=''):
    """Your own BGE-M3 worker: POST /embed {texts:[...]}, reply {vectors:[[...],...]}.
    Never forward texts to an untrusted endpoint.
    """
    if not url.startswith('https://') and not url.startswith('http://localhost'):
        raise ValueError('BGE-M3 worker must use HTTPS')
    headers={'Authorization':'Bearer '+token} if token else {}
    response=requests.post(url.rstrip('/')+'/embed',json={'texts':texts},headers=headers,timeout=120)
    response.raise_for_status()
    data=response.json().get('vectors',[])
    if len(data)!=len(texts): raise ValueError('BGE worker returned incorrect vector count')
    return data


def embed_texts(provider,texts,client=None,bge_url='',bge_token=''):
    if provider=='gemini2':
        if client is None: raise ValueError('Gemini client not configured')
        return embed_gemini(client,texts)
    if provider=='bge_m3':
        if not bge_url: raise ValueError('Configure BGE_M3_URL for BGE-M3 embeddings')
        return embed_bge_endpoint(texts,bge_url,bge_token)
    raise ValueError('Unknown provider')


def index_batch(store,provider,chunks,client=None,bge_url='',bge_token='',limit=12):
    """Index ONLY new/changed chunks, with atomic shard-first manifest-last writes."""
    meta,seen=load_vectors(store,provider)
    todo=[c for c in chunks if (c.get('kn') or c.get('en')) and chunk_id(c) not in seen]
    batch=todo[:max(1,min(limit,24))]
    if not batch: return {'indexed':0,'remaining':0,'total':len(seen),'provider':provider}
    vectors=embed_texts(provider,[embedding_text(c) for c in batch],client,bge_url,bge_token)
    packed=[]
    for c,v in zip(batch,vectors):
        encoded,dim=_pack(v)
        if meta['dim'] is not None and meta['dim']!=dim: raise ValueError('Embedding dimension changed: use a separate model/version index')
        meta['dim']=dim
        packed.append({'id':chunk_id(c),'v':encoded,'docid':c.get('docid',''),'page':c.get('page'), 'filename':c.get('filename','')})
    name='shard_%06d.json'%len(meta['shards'])
    _write(store,_path(provider)+'/'+name,{'records':packed},f'V7 {provider} checkpoint: {len(packed)} vectors')
    meta['shards'].append(name)
    meta['count']=meta.get('count',0)+len(packed)
    _write(store,_path(provider)+'/manifest.json',meta,f'V7 {provider} manifest checkpoint')
    remaining=len(todo)-len(batch)
    return {'indexed':len(batch),'remaining':remaining,'total':meta['count'],'provider':provider}


def semantic_search(query,provider,chunks,store,client=None,bge_url='',bge_token='',k=12):
    """Only match current source fingerprints. Returns score+source rows, not hallucinated answers."""
    meta,stored=load_vectors(store,provider)
    available=[(c,stored[chunk_id(c)]) for c in chunks if chunk_id(c) in stored]
    if not available:return [],{'indexed':0,'eligible':len(chunks),'coverage':0.0}
    q=embed_texts(provider,[query],client,bge_url,bge_token)[0]
    q=np.asarray(q,dtype=np.float32)
    if len(q)!=meta['dim']: raise ValueError('Query vector dimension differs from stored document vectors')
    q=q/(np.linalg.norm(q)+1e-12)
    scored=[]
    for c,v in available:
        try: score=float(np.dot(q,_unpack(v['v'],meta['dim'])))
        except Exception: continue
        scored.append((score,c))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [{'score':round(s,4),'source':c} for s,c in scored[:k]],{'indexed':len(available),'eligible':len(chunks),'coverage':round(len(available)/max(1,len(chunks)),4)}


def combine_ranked(keyword_sources, semantic_results, limit=12):
    """Reciprocal rank fusion: no uncalibrated blending of cosine and BM25."""
    allrows={}; weights={}
    for rank,c in enumerate(keyword_sources):
        key=chunk_id(c);allrows[key]=c;weights[key]=weights.get(key,0)+1/(60+rank+1)
    for rank,r in enumerate(semantic_results):
        c=r['source'];key=chunk_id(c);allrows[key]=c;weights[key]=weights.get(key,0)+1/(60+rank+1)
    ids=sorted(weights,key=weights.get,reverse=True)[:limit]
    return [allrows[i] for i in ids]
