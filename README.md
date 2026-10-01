# 🎓 College AI Assistant

A LangGraph-powered RAG assistant that answers college
questions using official academic and fee documents.

## Features

- Academic RAG
- Fee RAG
- Parallel RAG
- Query classification
- Follow-up question rewriting
- Programme-aware answers
- Streamlit interface
- FAISS vector search
- HuggingFace embeddings
- Groq LLMs

## Architecture

User
 ↓
Query Rewriter
 ↓
Classifier
 ↓
Academic / Fee / Both / General
 ↓
RAG
 ↓
Response Generator
 ↓
Answer

## Installation

...

## Environment Variables

...

## Run

streamlit run app.py

## Meta WhatsApp Webhook

Start the webhook server separately from Streamlit:

```bash
uvicorn webhook:app --host 0.0.0.0 --port 8000
```

Configure Meta with the public HTTPS callback URL `https://<your-domain>/webhook`.
The verification token is stored in the gitignored `.env.webhook` file as
`META_VERIFY_TOKEN`; alternatively, set that environment variable directly.
Meta's GET verification challenge is validated at `/webhook`, and POST event
payloads are acknowledged with HTTP 200.