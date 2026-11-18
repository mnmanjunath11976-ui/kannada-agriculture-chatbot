# Kannada Agriculture Knowledge Hub V7 — Neural Embeddings + Translation

**Upgrade to existing PUBLIC app repository** `mnmanjunath11976-ui/kannada-agriculture-chatbot`.

## What is genuinely implemented

- Everything from V6: multi-format upload, double-page landscape detection, page-wise Kannada OCR checkpoints, source PDF backup, English translation storage, local BM25/TF-IDF retrieval, citations, catalogue and voice.
- **Real Gemini Embedding 2** indexing: `client.models.embed_content(model="gemini-embedding-2", contents=text)`; never supplies the obsolete `task_type` field. The Gemini API key in existing Streamlit Secrets is reused. Indexes are saved in **private** `knowledge/v3_semantic/gemini2/` independent of `knowledge/v2`.
- **Real BGE-M3** indexing and query through an **optional dedicated embedding worker**. This package includes `bge_worker.py`, using `SentenceTransformer('BAAI/bge-m3')`, with its own `requirements-worker.txt` and bearer-token authentication. Install the worker on a machine with sufficient RAM/GPU; **do not deploy the worker into free Streamlit Cloud**. The Streamlit app sends texts only to the private worker URL you configure.
- Separate model spaces, separate versioned indexes. No mixing 1,024-dimensional BGE-M3 vectors with Gemini embeddings.
- Reciprocal rank fusion of keyword/BM25 and vector search. If the neural index is empty, the provider is down or Gemini returns 429/503, local retrieval stays active and private OCR data is unchanged.
- Checkpointed embedding batches of **12 passages per click**, deduplicated with a source/text SHA-256; changed chunks are reindexed but earlier vectors are not deleted automatically.
- English translation options: **Gemini** (default), **Google Cloud Translation Basic API** (`TRANSLATION_PROVIDER=google_cloud`), and an optional **IndicTrans2** HTTPS endpoint (`TRANSLATION_PROVIDER=indictrans2`). The packaged app implements the IndicTrans2 endpoint **client**, not a hosted translation server. Translation failures leave the original Kannada untouched and the translation pending.
- Existing V6 source records remain in `knowledge/v2`, untouched. Indexing is **not** automatic in the background; you explicitly trigger batches in the sidebar.

## Upload to GitHub

Upload to your existing **PUBLIC chatbot repository root**:

- `app.py` (replace)
- `retrieval.py` (replace)
- `catalogue.py` (replace)
- `document_ingest.py` (replace)
- `spread_ocr.py` (replace)
- `semantic_v7.py` (**new**)
- `requirements.txt` (replace)
- `packages.txt` (retain/replace)

Do **not** upload `bge_worker.py` to run on Streamlit. It is provided as a standalone service example for separate deployment. Never upload PDFs, Streamlit secrets, API keys, or access tokens to your PUBLIC repository.

## Streamlit Secrets

Existing configuration remains:

```toml
GEMINI_API_KEY = "YOUR_PRIVATE_KEY"
GITHUB_TOKEN = "YOUR_PRIVATE_REPO_SCOPED_TOKEN"
GITHUB_REPO = "mnmanjunath11976-ui/kannada-agri-knowledge-private"
APP_PASSWORD = "YOUR_PRIVATE_PASSWORD"
```

Optional Google Cloud Translation (requires the Cloud Translation API enabled, appropriate account/billing, and a restricted API key):

```toml
TRANSLATION_PROVIDER = "google_cloud"
GOOGLE_TRANSLATE_API_KEY = "YOUR_CLOUD_TRANSLATE_API_KEY"
```

Optional private BGE-M3 worker:

```toml
BGE_M3_URL = "https://YOUR-PRIVATE-EMBEDDING-WORKER"
BGE_M3_TOKEN = "MATCH_THE_WORKER_BEARER_TOKEN"
```

Run the `bge_worker.py` on a suitable machine behind authenticated HTTPS with `BGE_WORKER_TOKEN` set in that worker environment. `requirements-worker.txt` is separate from the free Streamlit requirements.

Optional IndicTrans2 translation endpoint (you must host/configure the translator separately):

```toml
TRANSLATION_PROVIDER = "indictrans2"
INDICTRANS2_URL = "https://YOUR-PRIVATE-TRANSLATION-WORKER"
INDICTRANS2_TOKEN = "YOUR_PRIVATE_WORKER_TOKEN"
```

## V7 workflow

1. Deploy V7; confirm previous saved document/page counts. The old private `knowledge/v2` records stay unchanged.
2. In sidebar **V7 neural embeddings**, select **Gemini Embedding 2** (works using your current API key, if your quota allows).
3. Click **Index next 12 passages**. Repeat as allowed by quota. Status shows a running count. Existing embeddings are reused on later visits. High-volume indexing will take many batches and can consume API quota.
4. Check **Use neural search for questions**. Ask the groundnut fertilizer question; open `Search details` and check whether `v7_semantic` shows indexed and eligible counts, and then open the exact source passages.
5. If Gemini is exhausted, BM25/TF-IDF search continues. BGE-M3 is an independent index option when your separate worker is connected.
6. To translate saved pages, choose your translation provider in Secrets, and use existing **Add / resume English translations**. Numeric discrepancies remain reviewable. Translation and OCR are not automatically error-free.

## Important limits / what is not claimed

- **No real OCR, translation, Gemini embedding or BGE-M3 worker call was performed in this build environment** because your API key, private GitHub data and separate worker are not available here. Python syntax and local synthetic tests were run; V7 needs actual live verification.
- A Google `429` requires available quota/billing, and a `503` can still occur. Failure is reported rather than concealed.
- Calling a model from a private worker still requires compute resources and deployment. The package does not provision a GPU server.
- No translator guarantees accurate technical tables. Verify crop, pesticide name, measurements, date, decimal points and all doses from original PDF images.
- Vector storage in private GitHub is suitable for this migration; for larger libraries use object storage or a vector DB. A large library takes many indexed batches.
- `gemini-embedding-2` can embed text directly; we intentionally embed **verified Kannada/English text passages**, not raw PDFs, to retain exact page and table citations.
- Current Gemini neural indexing and query calls still use API quota. The local BGE worker avoids Gemini embedding quota, but requires your own compute.

## Tests included

Run `python test_v7.py` for deterministic tests of embedding storage, provider isolation, changed passage re-indexing, semantic retrieval and quota-failure checkpoint safety.
