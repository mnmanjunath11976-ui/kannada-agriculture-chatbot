"""Deterministic collection-wide crop inventory for Kannada/English POP documents.
Read-only: scans existing indexed passages, makes no API calls or storage changes.
This is a crop *mention* inventory, NOT proof of official completeness.
"""
import re
from collections import defaultdict

# Canonical English, Kannada display, conservative known crop aliases.
CROPS = [
 ('Groundnut','ನೆಲಗಡಲೆ',['groundnut','peanut','ನೆಲಗಡಲೆ','ಕಡಲೆಕಾಯಿ','ಶೇಂಗಾ']),
 ('Paddy / Rice','ಭತ್ತ',['paddy','rice','ಭತ್ತ']),
 ('Wheat','ಗೋಧಿ',['wheat','ಗೋಧಿ']),
 ('Maize','ಮೆಕ್ಕೆಜೋಳ',['maize','corn','ಮೆಕ್ಕೆಜೋಳ']),
 ('Sorghum / Jowar','ಜೋಳ',['sorghum','jowar','ಜೋಳ']),
 ('Pearl millet / Bajra','ಸಜ್ಜೆ',['pearl millet','bajra','ಸಜ್ಜೆ']),
 ('Finger millet / Ragi','ರಾಗಿ',['finger millet','ragi','ರಾಗಿ']),
 ('Foxtail millet','ನವಣೆ',['foxtail millet','ನವಣೆ']),
 ('Little millet','ಸಾಮೆ',['little millet','ಸಾಮೆ']),
 ('Kodo millet','ಹಾರಕ',['kodo millet','ಹಾರಕ']),
 ('Barnyard millet','ಊದಲು',['barnyard millet','ಊದಲು']),
 ('Proso millet','ಬರಗು',['proso millet','ಬರಗು']),
 ('Black gram','ಉದ್ದು',['black gram','blackgram','urad','ಉದ್ದು','ಉದ್ದಿನ']),
 ('Green gram','ಹೆಸರುಕಾಳು',['green gram','greengram','mung','moong','ಹೆಸರುಕಾಳು','ಹೆಸರು ಬೆಳೆ']),
 ('Red gram / Pigeon pea','ತೊಗರಿ',['red gram','pigeon pea','pigeonpea','arhar','ತೊಗರಿ']),
 ('Bengal gram / Chickpea','ಕಡಲೆ',['bengal gram','chickpea','chana','ಕಡಲೆಕಾಳು','ಕಡಲೆ ಬೆಳೆ']),
 ('Horse gram','ಹುರಳಿ',['horse gram','ಹುರಳಿ']),
 ('Cowpea','ಅಲಸಂದೆ',['cowpea','ಅಲಸಂದೆ']),
 ('Field bean','ಅವರೆ',['field bean','ಅವರೆ']),
 ('Soybean','ಸೋಯಾಬೀನ್',['soybean','soya bean','ಸೋಯಾಬೀನ್','ಸೋಯಾ ಅವರೆ']),
 ('Sunflower','ಸೂರ್ಯಕಾಂತಿ',['sunflower','ಸೂರ್ಯಕಾಂತಿ']),
 ('Safflower','ಕುಸುಬೆ',['safflower','ಕುಸುಬೆ']),
 ('Sesame','ಎಳ್ಳು',['sesame','gingelly','ಎಳ್ಳು']),
 ('Niger','ಹುಚ್ಚೆಳ್ಳು',['niger seed','niger crop','ಹುಚ್ಚೆಳ್ಳು']),
 ('Mustard','ಸಾಸಿವೆ',['mustard','ಸಾಸಿವೆ']),
 ('Castor','ಹರಳು',['castor','ಹರಳು']),
 ('Sugarcane','ಕಬ್ಬು',['sugarcane','sugar cane','ಕಬ್ಬು']),
 ('Cotton','ಹತ್ತಿ',['cotton','ಹತ್ತಿ']),
 ('Tobacco','ತಂಬಾಕು',['tobacco','ತಂಬಾಕು']),
 ('Chilli','ಮೆಣಸಿನಕಾಯಿ',['chilli','chili','chilli pepper','ಮೆಣಸಿನಕಾಯಿ','ಮೆಣಸಿನಕಾಯಿಯ']),
 ('Onion','ಈರುಳ್ಳಿ',['onion','ಈರುಳ್ಳಿ']),
 ('Garlic','ಬೆಳ್ಳುಳ್ಳಿ',['garlic','ಬೆಳ್ಳುಳ್ಳಿ']),
 ('Tomato','ಟೊಮೇಟೊ',['tomato','tomatoe','ಟೊಮೇಟೊ','ಟೊಮ್ಯಾಟೊ']),
 ('Potato','ಆಲೂಗಡ್ಡೆ',['potato','ಆಲೂಗಡ್ಡೆ']),
 ('Brinjal / Eggplant','ಬದನೆಕಾಯಿ',['brinjal','eggplant','aubergine','ಬದನೆಕಾಯಿ']),
 ('Okra','ಬೆಂಡೆಕಾಯಿ',['okra','lady finger','ladies finger','ಬೆಂಡೆಕಾಯಿ']),
 ('Cabbage','ಎಲೆಕೋಸು',['cabbage','ಎಲೆಕೋಸು']),
 ('Cauliflower','ಹೂಕೋಸು',['cauliflower','ಹೂಕೋಸು']),
 ('Carrot','ಕ್ಯಾರೆಟ್',['carrot','ಕ್ಯಾರೆಟ್','ಗಜ್ಜರಿ']),
 ('Radish','ಮೂಲಂಗಿ',['radish','ಮೂಲಂಗಿ']),
 ('Cucumber','ಸೌತೆಕಾಯಿ',['cucumber','ಸೌತೆಕಾಯಿ']),
 ('Pumpkin','ಕುಂಬಳಕಾಯಿ',['pumpkin','ಕುಂಬಳಕಾಯಿ']),
 ('Bitter gourd','ಹಾಗಲಕಾಯಿ',['bitter gourd','bittergourd','ಹಾಗಲಕಾಯಿ']),
 ('Bottle gourd','ಸೊರೆಕಾಯಿ',['bottle gourd','ಸೊರೆಕಾಯಿ']),
 ('Ridge gourd','ಹೀರೆಕಾಯಿ',['ridge gourd','ಹೀರೆಕಾಯಿ']),
 ('Turmeric','ಅರಿಶಿನ',['turmeric','ಅರಿಶಿನ']),
 ('Ginger','ಶುಂಠಿ',['ginger','ಶುಂಠಿ']),
 ('Coriander','ಕೊತ್ತಂಬರಿ',['coriander','ಕೊತ್ತಂಬರಿ']),
 ('Cumin','ಜೀರಿಗೆ',['cumin','ಜೀರಿಗೆ']),
 ('Fenugreek','ಮೆಂತ್ಯೆ',['fenugreek','ಮೆಂತ್ಯೆ']),
 ('Banana','ಬಾಳೆ',['banana','ಬಾಳೆ']),
 ('Mango','ಮಾವು',['mango','ಮಾವು']),
 ('Coconut','ತೆಂಗು',['coconut','ತೆಂಗು']),
 ('Arecanut','ಅಡಿಕೆ',['arecanut','areca nut','ಅಡಿಕೆ']),
 ('Coffee','ಕಾಫಿ',['coffee','ಕಾಫಿ']),
 ('Tea','ಚಹಾ',['tea crop','tea cultivation','ಚಹಾ ಬೆಳೆ']),
 ('Grapes','ದ್ರಾಕ್ಷಿ',['grapes','grape','ದ್ರಾಕ್ಷಿ']),
 ('Pomegranate','ದಾಳಿಂಬೆ',['pomegranate','ದಾಳಿಂಬೆ']),
 ('Papaya','ಪಪ್ಪಾಯಿ',['papaya','ಪಪ್ಪಾಯಿ']),
 ('Guava','ಪೇರಲ',['guava','ಪೇರಲ']),
 ('Citrus / Orange','ಕಿತ್ತಳೆ',['orange','citrus','ಕಿತ್ತಳೆ']),
 ('Watermelon','ಕಲ್ಲಂಗಡಿ',['watermelon','ಕಲ್ಲಂಗಡಿ']),
 ('Muskmelon','ಕರಬೂಜ',['muskmelon','muskmelon','ಕರಬೂಜ']),
]

def is_catalogue_question(question):
    q = (question or '').casefold().strip()
    if not q:
        return False
    # Only collection-wide enumerations, not requests about one crop.
    en = bool(re.search(r'\b(list|names|enumerate|how many|number of|count|catalogue|catalog|inventory|all)\b', q)) and bool(re.search(r'\b(crop|crops|plants|commodities)\b', q))
    kn = ('ಬೆಳೆ' in q or 'ಕೃಷಿ ಬೆಳೆ' in q) and any(x in q for x in ('ಪಟ್ಟಿ','ಎಷ್ಟು','ಎಲ್ಲಾ','ಯಾವುವು','ಹೆಸರು','ಸಂಖ್ಯೆ','ವಿವರ'))
    return en or kn


def _has_alias(text, alias):
    term = alias.casefold()
    if re.search(r'[a-z]', term):
        return bool(re.search(r'(?<![a-z])'+re.escape(term)+r'(?![a-z])', text))
    # Kannada suffix forms are common; prevent very short aliases matching longer unrelated words.
    return term in text


def catalogue_inventory(chunks, max_evidence=2):
    """Return unique matched crop names plus evidence from all available chunks."""
    hits = defaultdict(lambda: {'pages': set(), 'evidence': []})
    scanned_pages = set()
    for c in chunks:
        file = str(c.get('filename', 'Unknown'))
        page = c.get('page', '?')
        key = (str(c.get('docid') or file), str(page))
        scanned_pages.add(key)
        body = ((c.get('kn') or '') + '\n' + (c.get('en') or '')).casefold().replace('\u200c','').replace('\u200d','')
        if not body.strip():
            continue
        for name, kannada, aliases in CROPS:
            if not any(_has_alias(body, a.replace('\u200c','').replace('\u200d','')) for a in aliases):
                continue
            rec = hits[name]
            if key not in rec['pages']:
                rec['pages'].add(key)
                if len(rec['evidence']) < max_evidence:
                    rec['evidence'].append({'filename':file,'page':page,'kn':c.get('kn',''),'en':c.get('en',''),'docid':c.get('docid','')})
    rows=[]
    for name, kannada, _ in CROPS:
        if name in hits:
            rec=hits[name]
            rows.append({'name':name,'kannada':kannada,'page_count':len(rec['pages']),'evidence':rec['evidence']})
    rows.sort(key=lambda r:r['name'].casefold())
    return rows, len(scanned_pages)


def catalogue_answer(rows, n_pages, n_docs, kannada=False):
    if not rows:
        return ('ದಾಖಲೆಗಳ ಹುಡುಕಬಹುದಾದ ಪಠ್ಯದಲ್ಲಿ ತಿಳಿದಿರುವ ಬೆಳೆ ಹೆಸರುಗಳನ್ನು ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. OCR ಮತ್ತು ದಾಖಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.' if kannada else 'No crop names were detected in searchable text. Check OCR and source documents.')
    if kannada:
        intro=f'**ಸಂಗ್ರಹಿತ ದಾಖಲೆಗಳ ಪಠ್ಯದಲ್ಲಿ ಗುರುತಿಸಿದ {len(rows)} ವಿಭಿನ್ನ ಬೆಳೆ ಹೆಸರುಗಳು** ({n_docs} ದಾಖಲೆಗಳು; {n_pages} ಹುಡುಕಬಹುದಾದ ಪುಟಗಳನ್ನು ಪರಿಶೀಲಿಸಲಾಗಿದೆ):'
        note='**ಗಮನಿಸಿ:** ಇದು OCR/ಅನುವಾದದಲ್ಲಿ ಗುರುತಿಸಿದ ಬೆಳೆ ಹೆಸರುಗಳ ಪಟ್ಟಿ ಮಾತ್ರ. ಎಲ್ಲಾ ಬೆಳೆಗಳ ಅಧಿಕೃತ ಅಥವಾ ಸಂಪೂರ್ಣ ಪಟ್ಟಿ ಎಂದು ಪರಿಗಣಿಸಬೇಡಿ. ಮೂಲ ಪುಟಗಳನ್ನು ಕೆಳಗೆ ಪರಿಶೀಲಿಸಿ.'
    else:
        intro=f'**{len(rows)} distinct crop names detected in your saved documents** (scanned {n_pages} searchable pages across {n_docs} documents):'
        note='**Note:** This is a detected-name inventory, not a certified complete list of crops. OCR gaps, translation gaps and unfamiliar crop names can cause omissions. Check the source pages below.'
    lines=[intro,'']
    for i,r in enumerate(rows,1):
        lines.append(f"{i}. **{r['name']} — {r['kannada']}** · {r['page_count']} page(s)")
    lines.extend(['',note])
    return '\n'.join(lines)
