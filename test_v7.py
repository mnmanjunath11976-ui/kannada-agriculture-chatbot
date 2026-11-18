import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import semantic_v7 as s
class MemoryStore:
    def __init__(self):self.db={}
    def get(self,path):return (self.db.get(path),None)
    def put(self,path,data,note=''):self.db[path]=data

def fake(provider,texts,client=None,bge_url='',bge_token=''):
    out=[]
    for t in texts:
        vec=np.zeros(1024,dtype=np.float32)
        vec[0]=1.0 if ('ನೆಲಗಡಲೆ' in t or 'groundnut' in t.lower()) else 0
        vec[1]=1.0 if ('ಜಿಪ್ಸಂ' in t or 'gypsum' in t.lower()) else 0
        vec[2]=1.0 if ('rice' in t.lower()) else 0
        if not vec.any():vec[3]=1
        out.append(vec.tolist())
    return out
s.embed_texts=fake
store=MemoryStore()
chunks=[{'docid':'one','page':1,'filename':'original.pdf','source_location':'PDF p3 left','kn':'ನೆಲಗಡಲೆ ಬೆಳೆಗೆ ಜಿಪ್ಸಂ ಶಿಫಾರಸು','en':'Groundnut gypsum recommendation'}, {'docid':'one','page':2,'filename':'original.pdf','kn':'ಭತ್ತ', 'en':'Rice'}]
assert s.index_batch(store,'bge_m3',chunks,limit=12)['indexed']==2
assert s.index_batch(store,'bge_m3',chunks,limit=12)['indexed']==0
assert s.load_manifest(store,'bge_m3')['count']==2
result,stats=s.semantic_search('ನೆಲಗಡಲೆ ಜಿಪ್ಸಂ','bge_m3',chunks,store)
assert result[0]['source']['page']==1
assert stats['coverage']==1.0
assert s.combine_ranked([chunks[1]],[{'source':chunks[0],'score':0.8}],2)
changed=[dict(chunks[0],kn='NEW text'),chunks[1]]
assert s.index_batch(store,'bge_m3',changed)['indexed']==1
assert s.semantic_search('rice','bge_m3',changed,store)[1]['coverage']==1.0
assert s.load_manifest(store,'gemini2')['count']==0
# A failed embedding must not create an index checkpoint.
def failing(*args,**kwargs):raise RuntimeError('429 resource exhausted')
s.embed_texts=failing
before=set(store.db)
try:s.index_batch(store,'gemini2',chunks)
except RuntimeError:pass
else:raise AssertionError('Missing quota failure')
assert before==set(store.db)
print('PASS: separate providers, persisted shards, semantic retrieval, stable hashes, changed text and 429 safety')
