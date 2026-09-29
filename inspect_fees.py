# pip install pdfplumber
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF = "fee_structure.pdf"

print("=== 1. What your app indexes today (PyPDFLoader) ===")
pages = PyPDFLoader(PDF).load()
for d in pages:
    print(f"--- page {d.metadata.get('page')} | {len(d.page_content)} chars ---")
    print(repr(d.page_content[:1500]))

print("\n=== 2. What pdfplumber sees as tables ===")
with pdfplumber.open(PDF) as pdf:
    for i, page in enumerate(pdf.pages, 1):
        tables = page.extract_tables()
        print(f"--- page {i}: {len(tables)} table(s) ---")
        for t in tables:
            for row in t:
                print(row)
            print()

print("\n=== 3. What your retriever returns today ===")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100
).split_documents(pages)
store = FAISS.from_documents(
    chunks,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
retriever = store.as_retriever(search_kwargs={"k": 4})
for q in ["BBA tuition fee", "BCA tuition fee", "late payment charge"]:
    print(f"\n>>> {q}")
    for i, d in enumerate(retriever.invoke(q), 1):
        print(f"[{i}] p{d.metadata.get('page')}: {d.page_content[:300]!r}")
        