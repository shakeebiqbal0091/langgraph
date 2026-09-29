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