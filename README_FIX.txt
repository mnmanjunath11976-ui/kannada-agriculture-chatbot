Kannada Agriculture Knowledge Hub V2 — Sidebar Fix

Fixes: Streamlit displayed its internal DeltaGenerator object/source text in the sidebar due to a conditional expression. The condition is now a normal if/else block.

Upload app.py and requirements.txt to the existing PUBLIC code repository. Keep your GEMINI_API_KEY, GITHUB_TOKEN, GITHUB_REPO, and APP_PASSWORD only in Streamlit Secrets. Do not upload PDF documents to the public code repository.

Once updated, unlock your app. 0 saved documents is expected before the first upload. Use Add PDF to knowledge -> Save to knowledge base. Then reload and verify a saved document remains visible.

This patch passes Python syntax validation; live GitHub write and OCR have not been verified with your account.
