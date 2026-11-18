"""Resumable, spread-aware Kannada PDF ingestion. No Streamlit or API dependencies.

This is a non-destructive extension to knowledge/v2. All original page and
side references are preserved; an OCR failure never becomes searchable text.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
import pymupdf

VERSION = "spread-ocr-v1"
PART_SIZE = 650_000  # below the existing GithubStore.put 850 KB single-write ceiling
BATCH_SIZE = 4
MIN_KANNADA = 0.14


def unicode_quality(text: str) -> dict:
    kn = sum('\u0c80' <= c <= '\u0cff' for c in text)
    alpha = sum(c.isalpha() for c in text)
    # The PDF uses legacy non-Unicode Kannada glyphs. Do not index gibberish.
    legacy = bool(re.search(r'(?:À|Á|Ä|¸|¨|ª|£|¢|Æ|¥|gï|PÀ)', text))
    quality = kn / max(alpha, 1)
    good = kn >= 30 and quality >= MIN_KANNADA and not (legacy and quality < 0.50)
    return {"kannada_chars": kn, "letter_count": alpha, "kannada_ratio": round(quality, 4), "legacy_suspected": legacy, "usable": good}


def locations(pdf: pymupdf.Document, layout="auto"):
    """Return logical pages (PDF index, L/R/full, printed index not assumed)."""
    out=[]
    for i,p in enumerate(pdf):
        spread = layout == 'two-page spread' or (layout == 'auto' and p.rect.width / max(1,p.rect.height) > 1.20)
        sides = ('left','right') if spread else ('full',)
        for side in sides:
            out.append({"pdf_page":i+1,"side":side,"number":len(out)+1,
                        "source_location": f"PDF page {i+1} · {side} half" if spread else f"PDF page {i+1} · full page"})
    return out


def clip_for(page, side):
    r=page.rect
    if side == 'left': return pymupdf.Rect(r.x0,r.y0,(r.x0+r.x1)/2,r.y1)
    if side == 'right': return pymupdf.Rect((r.x0+r.x1)/2,r.y0,r.x1,r.y1)
    return r


def native_text(page, side):
    return page.get_text('text', clip=clip_for(page,side), sort=True).strip()


def ocr_image(page, side, dpi=200):
    matrix=pymupdf.Matrix(dpi/72,dpi/72)
    pix=page.get_pixmap(matrix=matrix, clip=clip_for(page,side), alpha=False)
    # Avoid oversized per-part GitHub records; no storage of image needed,
    # because the original PDF is checkpointed before any OCR.
    return pix.tobytes('jpeg', jpg_quality=82)


def document_id(pdf_data):
    return hashlib.sha256(pdf_data+VERSION.encode()).hexdigest()[:24]


def write_json(store,path,obj,note):
    store.put(path,json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode('utf-8'),note)


def read_json(store,path,default=None):
    blob,_=store.get(path)
    return json.loads(blob.decode('utf-8')) if blob else default


def start_document(store,root,name,pdf_data,layout='auto'):
    docid=document_id(pdf_data)
    prefix=f'{root}/docs/{docid}'
    meta=read_json(store,f'{prefix}/meta.json')
    if meta:
        return meta,False
    pdf=pymupdf.open(stream=pdf_data,filetype='pdf')
    logical=locations(pdf,layout)
    pdf.close()
    # Checkpoint original PDF before registering it, so pending OCR can be
    # retried without requiring another upload, even after app restart.
    parts=math.ceil(len(pdf_data)/PART_SIZE)
    for i in range(parts):
        store.put(f'{prefix}/original/part_{i:04d}.bin',
                  pdf_data[i*PART_SIZE:(i+1)*PART_SIZE],f'Backup source PDF {name}, part {i+1}/{parts}')
    meta={"id":docid,"name":name,"total_pages":len(logical),"stored_pages":0,
          "pdf_pages":len(set(x['pdf_page'] for x in logical)), "original_parts":parts,
          "source_sha256":hashlib.sha256(pdf_data).hexdigest(), "source_bytes":len(pdf_data),
          "layout":layout,"batches":0,"status":"processing", "pipeline_version":VERSION,
          "added_utc":datetime.now(timezone.utc).isoformat()}
    write_json(store,f'{prefix}/meta.json',meta,'Create spread-aware PDF record')
    manifest=read_json(store,f'{root}/manifest.json',{"documents":[]})
    if not any(d.get('id')==docid for d in manifest.get('documents',[])):
        manifest['documents'].append({'id':docid,'name':name})
        write_json(store,f'{root}/manifest.json',manifest,'Register spread-aware PDF')
    return meta,True


def load_original(store,root,meta):
    prefix=f"{root}/docs/{meta['id']}/original"
    chunks=[]
    for i in range(meta['original_parts']):
        blob,_=store.get(f'{prefix}/part_{i:04d}.bin')
        if not blob: raise RuntimeError(f'Missing saved PDF part {i+1}/{meta["original_parts"]}')
        chunks.append(blob)
    data=b''.join(chunks)
    if hashlib.sha256(data).hexdigest()!=meta['source_sha256']:
        raise RuntimeError('Original PDF SHA-256 checksum mismatch')
    return data


def make_page(pdf,loc,ocr,do_translate=None):
    p=pdf[loc['pdf_page']-1]
    extracted=native_text(p,loc['side'])
    quality=unicode_quality(extracted)
    method='Unicode PDF text' if quality['usable'] else 'OCR pending'
    text=extracted if quality['usable'] else ''
    issue=''
    if not quality['usable']:
        try:
            data=ocr_image(p,loc['side'])
            result=(ocr(data,loc) or '').strip()
            if result and sum('\u0c80'<=c<='\u0cff' for c in result)>20:
                text=result
                method='Image OCR (needs review)'
            else:
                issue='OCR returned no readable Kannada; verify visually or retry'
        except Exception as e:
            issue=f'OCR API failed: {type(e).__name__}'
    en=''
    translation_pending=bool(text)
    if text and do_translate:
        try:
            en=do_translate(text) or ''
        except Exception:
            pass
        translation_pending=not bool(en)
    return {**loc,'kn':text,'en':en,'method':method,'translation_pending':translation_pending,
            'ocr_status': 'pending' if not text else ('review' if method.startswith('Image OCR') else 'native'),
            'ocr_issue':issue,'native_quality':quality,
            'source_location':loc['source_location']}


def page_batches(pages):
    for i in range(0,len(pages),BATCH_SIZE):
        yield i//BATCH_SIZE,pages[i:i+BATCH_SIZE]


def process_next(store,root,meta,ocr,do_translate=None,max_logical_pages=6):
    data=load_original(store,root,meta)
    pdf=pymupdf.open(stream=data,filetype='pdf')
    logical=locations(pdf,meta.get('layout','auto'))
    start=meta['stored_pages']
    end=min(start+max_logical_pages,len(logical))
    result=[]
    try:
        for i in range(start,end):
            result.append(make_page(pdf,logical[i],ocr,do_translate))
    finally:
        pdf.close()
    # Append partial last batch properly, so every page occupies its stable
    # logical number and no existing batch gets overwritten or duplicated.
    prefix=f"{root}/docs/{meta['id']}"
    for page in result:
        bnum=(page['number']-1)//BATCH_SIZE
        path=f'{prefix}/batch_{bnum:04d}.json'
        group=read_json(store,path,[]) or []
        group=[item for item in group if item['number']!=page['number']]
        group.append(page)
        group.sort(key=lambda x:x['number'])
        write_json(store,path,group,f'Save PDF logical page {page["number"]}')
    meta['stored_pages']=end
    meta['batches']=math.ceil(end/BATCH_SIZE)
    meta['status']='processing' if end<len(logical) else 'ready_with_review'
    write_json(store,f'{prefix}/meta.json',meta,'Checkpoint PDF OCR progress')
    return result


def retry_page(store,root,meta,logical_number,ocr,do_translate=None):
    if logical_number<1 or logical_number>meta['stored_pages']:
        raise ValueError('Logical page not yet processed')
    path=f"{root}/docs/{meta['id']}/batch_{(logical_number-1)//BATCH_SIZE:04d}.json"
    batch=read_json(store,path,[]) or []
    original=next((p for p in batch if p['number']==logical_number),None)
    if original is None: raise ValueError('Missing logical page record')
    data=load_original(store,root,meta)
    pdf=pymupdf.open(stream=data,filetype='pdf')
    try:
        loc=locations(pdf,meta.get('layout','auto'))[logical_number-1]
        new=make_page(pdf,loc,ocr,do_translate)
    finally:
        pdf.close()
    # Never replace valid OCR with failed OCR.
    if new['kn'] or not original.get('kn'):
        batch=[new if x['number']==logical_number else x for x in batch]
        write_json(store,path,batch,f'Retry logical page OCR {logical_number}')
    return new


def audit(pages):
    from collections import Counter
    c=Counter(p.get('ocr_status','unknown') for p in pages)
    return {"total_records":len(pages),"native":c['native'],"ocr_needs_review":c['review'],
            "pending":c['pending'],"english_missing":sum(1 for p in pages if p.get('kn') and not p.get('en')),
            "indexed_text_pages":sum(1 for p in pages if p.get('kn','').strip())}
