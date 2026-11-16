KANNADA AGRICULTURE RAG CHATBOT - VOICE UPDATE

Files to upload to your existing GitHub repository:
  1) app.py
  2) requirements.txt

GitHub repository:
https://github.com/mnmanjunath11976-ui/kannada-agriculture-chatbot

Updated features:
- Same multi-PDF Kannada / English question-answer chatbot
- Source document and page citations remain
- One-click spoken answer in Kannada / English (gTTS)
- Audio player + downloadable MP3 for each reply
- Optional slow speech
- Record voice questions from browser microphone, transcribed by Gemini
- Retains Gemini 3.8 Flash and 3.5 Flash-Lite fallback on 404
- Local TF-IDF retrieval remains available if Gemini embeddings hit quota

How to update:
1. Extract this ZIP (do not upload the ZIP itself).
2. Visit GitHub repo and use Add file > Upload files.
3. Upload both files, replacing the same filenames in the repository root.
4. Commit changes. Streamlit Cloud normally redeploys automatically.
5. Open your existing Streamlit app, provide your Gemini API key, and rebuild the PDF knowledge base if session was reset.
6. Ask a question; click 'Listen / ಉತ್ತರವನ್ನು ಕೇಳಿ' below its answer.
7. For voice questions, expand the microphone section, record, and click 'Transcribe & Ask'.

Important limitations:
- The old SadTalker Colab code animated a still picture using an existing approved MP3.
  It did not synthesize speech from text. This update creates NEW spoken audio
  for dynamic PDF answers but DOES NOT create a lip-synced presenter video and
  DOES NOT reproduce the exact voice in your approved MP3.
- Streamlit Community Cloud isn't suitable for local SadTalker GPU inference.
- gTTS requires an internet connection and the remote service may rate-limit.
- Speech recognition and answers consume Gemini API quota; the earlier 429
  quota error may still affect them. Spoken answer generation via gTTS does
  not itself use Gemini API embedding quota.
- The online gTTS service receives the answer text when you click Listen.
- Keep API keys out of your public GitHub repository. Revoke any exposed key.
