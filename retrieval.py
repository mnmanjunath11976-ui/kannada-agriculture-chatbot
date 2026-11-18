"""Offline Kannada/English hybrid retrieval. No API calls, no data writes.
Designed to load the existing knowledge/v2 saved-page records unchanged.
"""
import math
import re
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Include known agronomy synonyms AND bilingual agricultural terminology.
# Add vocabulary here without re-OCR or retranslating stored pages.
CONCEPTS = {
    "groundnut": ["groundnut", "peanut", "peanuts", "ನೆಲಗಡಲೆ", "ಕಡಲೆಕಾಯಿ", "ಶೇಂಗಾ", "ಶೇಂಗಾ ಬೆಳೆ"],
    "blackgram": ["black gram", "blackgram", "urad", "ಉದ್ದು", "ಉದ್ದಿನ", "ಮಾಷ್"],
    "greengram": ["green gram", "mung", "moong", "ಹೆಸರು", "ಹೆಸರುಕಾಳು"],
    "chickpea": ["chickpea", "bengal gram", "ಕಡಲೆಕಾಳು"],
    "paddy": ["paddy", "rice", "ಭತ್ತ", "ಅಕ್ಕಿ"],
    "maize": ["maize", "corn", "ಮೆಕ್ಕೆಜೋಳ"],
    "ragi": ["finger millet", "ragi", "ರಾಗಿ"],
    "fertilizer": ["fertilizer", "fertiliser", "nutrient", "manure", "ಗೊಬ್ಬರ", "ರಸಗೊಬ್ಬರ", "ಪೋಷಕಾಂಶ", "ಪೋಷಕ", "ರಸಗೊಬ್ಬರಗಳು"],
    "basal": ["basal dose", "basal application", "at sowing", "ಬಿತ್ತನೆ ಸಮಯದಲ್ಲಿ", "ಬಿತ್ತನೆ ವೇಳೆ", "ಮೂಲ ಗೊಬ್ಬರ"],
    "rhizobium": ["rhizobium", "ರೈಜೋಬಿಯಂ", "ರೈಜೋಬಿಯಮ್", "ರೈಜೋಬಿಯಮ್ ಸಂಸ್ಕೃತಿ"],
    "psb": ["psb", "phosphate solubilizing bacteria", "phosphorus solubilizing bacteria", "ಪಿಎಸ್‌ಬಿ", "ಪಿ ಎಸ್ ಬಿ", "ಪಿಎಸ್ಬಿ"],
    "gypsum": ["gypsum", "ಜಿಪ್ಸಂ", "ಜಿಪ್ಸಮ್"],
    "dose": ["dose", "dosage", "quantity", "rate", "recommendation", "recommended", "ಪ್ರಮಾಣ", "ಶಿಫಾರಸು", "ಎಷ್ಟು", "ಬಳಕೆ"],
    "seed": ["seed", "seed rate", "ಬೀಜ", "ಬಿತ್ತನೆ"],
    "disease": ["disease", "ರೋಗ", "ರೋಗಗಳು"],
    "pest": ["pest", "insect", "ಕೀಟ", "ಹುಳು"],
    "irrigation": ["irrigation", "watering", "ನೀರಾವರಿ", "ನೀರು"],
    "yield": ["yield", "harvest", "ಇಳುವರಿ", "ಕೊಯ್ಲು"],
}

def normalize(s):
    s = (s or "").lower().replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", s).strip()


def words(text):
    # Kannada block is U+0C80-U+0CFF. Preserve combining marks and digits.
    return re.findall(r"[\u0c80-\u0cff]+|[a-z]+|\d+(?:\.\d+)?", normalize(text))


def concept_hits(query):
    q = normalize(query)
    tokens = set(words(q))
    hits = []
    for concept, variants in CONCEPTS.items():
        for variant in variants:
            v = normalize(variant)
            # Match Kannada suffix-rich words by prefix; avoid substring collisions for very short forms.
            if (len(v) >= 4 and (v in q or any(t.startswith(v) for t in tokens))) or (v in tokens):
                hits.append(concept)
                break
    return hits


def expand(query):
    keys = concept_hits(query)
    additional = []
    for k in keys:
        additional += CONCEPTS[k]
    # Keep original question repeated to avoid its concepts being overwhelmed by synonyms.
    return normalize(query) + " " + normalize(query) + " " + " ".join(additional), keys


class HybridIndex:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.texts = [normalize(c.get("kn", "") + " " + c.get("en", "") + " " + c.get("filename", "")) for c in self.chunks]
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=70000, sublinear_tf=True, dtype=np.float32)
        self.char_matrix = self.char.fit_transform(self.texts)
        # Construct lightweight inverted index (BM25): exact term matching for agrochemical names and numbers.
        self.postings = defaultdict(list)
        self.dl = np.zeros(len(self.texts), dtype=np.float32)
        for i, text in enumerate(self.texts):
            counts = Counter(words(text))
            self.dl[i] = sum(counts.values())
            for term, count in counts.items():
                self.postings[term].append((i, count))
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0

    def search(self, question, k=7):
        if not self.chunks or not question.strip():
            return [], {"message": "No searchable text or question", "concepts": []}
        expanded, concepts = expand(question)
        qvec = self.char.transform([expanded])
        char_scores = (self.char_matrix @ qvec.T).toarray().reshape(-1)
        bm25 = np.zeros(len(self.chunks), dtype=np.float32)
        query_terms = Counter(words(expanded))
        N = len(self.chunks)
        for term, repeats in query_terms.items():
            items = self.postings.get(term)
            if not items:
                continue
            df = len(items)
            idf = math.log1p((N - df + .5)/(df + .5))
            for i, freq in items:
                bm25[i] += idf * (freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * self.dl[i] / max(1, self.avgdl)))) * min(repeats, 2)
        bm25 = bm25 / (bm25.max() + 1e-6) if bm25.max() > 0 else bm25
        # Boost matching target concepts without assuming they prove an answer.
        boosts = np.zeros(N, dtype=np.float32)
        for concept in concepts:
            variants = CONCEPTS[concept]
            for i, text in enumerate(self.texts):
                if any(normalize(v) in text for v in variants):
                    boosts[i] += 0.03
        scores = 0.55 * char_scores + 0.42 * bm25 + np.minimum(boosts, 0.12)
        ranked = np.argsort(-scores)
        found, per_page = [], Counter()
        # No hard 0.035 cutoff: inspect top results and signal weak matches explicitly.
        for idx in ranked:
            i = int(idx)
            if scores[i] <= 0.005:
                break
            c = self.chunks[i]
            page_key = (c.get("docid"), c.get("page"))
            if per_page[page_key] >= 2:
                continue
            found.append({**c, "score": round(float(scores[i]), 4)})
            per_page[page_key] += 1
            if len(found) >= k:
                break
        best = float(found[0]["score"]) if found else 0.0
        diagnostics = {
            "concepts": concepts,
            "top_score": round(best, 4),
            "quality": "weak" if best < 0.07 else "candidate passages (not verified facts)",
            "indexed_chunks": N,
            "matches": len(found),
            "mode": "offline BM25 + Kannada/English synonym expansion + character TF-IDF",
        }
        return found, diagnostics


def likely_document_question(question, history=None):
    s = normalize(question)
    if re.fullmatch(r"(hello|hi|hey|good morning|good evening|thank you|thanks|ನಮಸ್ಕಾರ|ಹಾಯ್|ಹಲೋ|ಧನ್ಯವಾದ)[! ?.]*", s):
        return False
    if concept_hits(s):
        return True
    cues = ("pdf", "document", "source", "page", "manual", "pop", "crop", "agri", "farm", "soil", "rain", "ಕೃಷಿ", "ಬೆಳೆ", "ದಾಖಲೆ", "ಪುಟ", "ಹೇಗೆ", "ಏನು")
    if any(cue in s for cue in cues):
        return True
    # Short follow-ups after a document conversation retain context.
    prior = (history or [])[-5:]
    return len(words(s)) <= 8 and any(h.get("routed_doc") for h in prior if h.get("role") == "assistant")


def contextual_query(question, history):
    s = normalize(question)
    history = history or []
    if concept_hits(s) and any(k in concept_hits(s) for k in ("groundnut", "blackgram", "greengram", "chickpea", "paddy", "maize", "ragi")):
        return question
    if len(words(s)) > 14:
        return question
    for item in reversed(history[-8:]):
        if item.get("role") == "user":
            previous = item.get("text", "")
            crop_hits = [k for k in concept_hits(previous) if k in ("groundnut", "blackgram", "greengram", "chickpea", "paddy", "maize", "ragi")]
            if crop_hits:
                return question + " " + " ".join(CONCEPTS[k][0] for k in crop_hits)
    return question
