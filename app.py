import os
import re
import operator
from typing import Annotated, TypedDict

import pdfplumber
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Create a .env file and add: GROQ_API_KEY=your_api_key"
    )
    st.stop()


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# 3. CONSTANTS
# ============================================================

ACADEMIC_PDF = "academics_handbook.pdf"
FEE_PDF = "fee_structure.pdf"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Normal generation models.
# If one of these is unavailable in your Groq account,
# replace it with a model returned by Groq's model list.
CLASSIFIER_MODEL = "openai/gpt-oss-20b"
RESPONSE_MODEL = "openai/gpt-oss-120b"


# ============================================================
# 4. SMALL TALK DETECTION
# ============================================================

SMALLTALK_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|"
    r"good morning|good afternoon|good evening|"
    r"how are you|how are you doing|"
    r"thanks|thank you|"
    r"bye|goodbye"
    r")\s*[!.?]*\s*$",
    re.IGNORECASE,
)


# ============================================================
# 5. LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    programme: str

    messages: Annotated[list, add_messages]

    query_type: str

    # operator.add allows academic and fee RAG branches
    # to write to the same state in parallel.
    retrieved_context: Annotated[list[str], operator.add]


# ============================================================
# 6. LLMs
# ============================================================

classifier_llm = ChatGroq(
    model=CLASSIFIER_MODEL,
    temperature=0,
    groq_api_key=GROQ_API_KEY,
)

response_llm = ChatGroq(
    model=RESPONSE_MODEL,
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
)


# ============================================================
# 7. LOAD FEE PDF
# ============================================================

def load_fee_documents(pdf_path: str) -> list[Document]:
    """
    Extract fee information from the PDF using pdfplumber.

    pdfplumber is used here instead of PyPDFLoader because
    fee structures often contain tables and formatted rows.
    """

    documents: list[Document] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                # Try extracting tables first.
                tables = page.extract_tables()

                if tables:

                    for table_number, table in enumerate(tables, start=1):

                        rows: list[str] = []

                        for row in table:

                            if not row:
                                continue

                            cleaned_row = [
                                str(cell).strip() if cell is not None else ""
                                for cell in row
                            ]

                            if any(cleaned_row):
                                rows.append(" | ".join(cleaned_row))

                        if rows:

                            table_text = "\n".join(rows)

                            documents.append(
                                Document(
                                    page_content=table_text,
                                    metadata={
                                        "source": pdf_path,
                                        "page": page_number,
                                        "type": "fee_table",
                                        "table": table_number,
                                    },
                                )
                            )

                # Also extract normal page text.
                page_text = page.extract_text()

                if page_text and page_text.strip():

                    documents.append(
                        Document(
                            page_content=page_text.strip(),
                            metadata={
                                "source": pdf_path,
                                "page": page_number,
                                "type": "fee_text",
                            },
                        )
                    )

    except Exception as exc:
        raise RuntimeError(
            f"Could not read fee PDF '{pdf_path}': {exc}"
        ) from exc

    if not documents:
        raise RuntimeError(
            f"No readable content was extracted from '{pdf_path}'."
        )

    return documents


# ============================================================
# 8. LOAD RAG RESOURCES
# ============================================================

@st.cache_resource
def load_resources():

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Text splitter
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    # --------------------------------------------------------
    # Academic documents
    # --------------------------------------------------------

    if not os.path.exists(ACADEMIC_PDF):
        raise FileNotFoundError(
            f"Academic PDF not found: {ACADEMIC_PDF}"
        )

    academic_loader = PyPDFLoader(ACADEMIC_PDF)

    academic_documents = academic_loader.load()

    academic_chunks = splitter.split_documents(
        academic_documents
    )

    academic_vectorstore = FAISS.from_documents(
        academic_chunks,
        embeddings,
    )

    academic_retriever = academic_vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # --------------------------------------------------------
    # Fee documents
    # --------------------------------------------------------

    if not os.path.exists(FEE_PDF):
        raise FileNotFoundError(
            f"Fee PDF not found: {FEE_PDF}"
        )

    # IMPORTANT:
    # Use the custom fee parser here.
    # This fixes the previous issue where
    # load_fee_documents() existed but was never used.

    fee_documents = load_fee_documents(FEE_PDF)

    fee_chunks = splitter.split_documents(
        fee_documents
    )

    fee_vectorstore = FAISS.from_documents(
        fee_chunks,
        embeddings,
    )

    fee_retriever = fee_vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    return academic_retriever, fee_retriever


# ============================================================
# 9. LOAD RESOURCES
# ============================================================

try:

    academic_retriever, fee_retriever = load_resources()

except Exception as exc:

    st.error("Failed to load the college documents.")

    st.exception(exc)

    st.stop()


# ============================================================
# 10. HELPER FUNCTIONS
# ============================================================

def get_latest_user_message(messages: list) -> str:
    """
    Return the latest HumanMessage from the conversation.
    """

    for message in reversed(messages):

        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def format_documents(documents: list[Document]) -> str:
    """
    Convert retrieved documents into readable context.
    """

    if not documents:
        return "No relevant information was found."

    formatted_parts: list[str] = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "unknown source",
        )

        page = document.metadata.get(
            "page",
            "unknown page",
        )

        formatted_parts.append(
            f"[Source {index} | Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_parts)


# ============================================================
# 11. REWRITE NODE
# ============================================================

def rewrite_node(state: State) -> dict:

    query = get_latest_user_message(
        state["messages"]
    )

    if not query:
        return {}

    # Do not waste an LLM call on simple greetings.
    if SMALLTALK_RE.match(query):
        return {}

    programme = state["programme"]

    rewrite_prompt = f"""
You are a query rewriting assistant for a college information system.

Student programme:
{programme}

Conversation:
{state["messages"]}

Rewrite the student's latest question into a clear,
standalone search query.

Rules:
- Preserve the student's original meaning.
- Resolve references such as "it", "that", "this", "they", etc.
  using the conversation.
- Include the student's programme when it is relevant.
- Do not answer the question.
- Return ONLY the rewritten question.
"""

    response = classifier_llm.invoke(
        rewrite_prompt
    )

    rewritten_query = str(response.content).strip()

    if not rewritten_query:
        return {}

    # Replace the latest user message with the rewritten query.
    return {
        "messages": [
            HumanMessage(content=rewritten_query)
        ]
    }


# ============================================================
# 12. CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:

    query = get_latest_user_message(
        state["messages"]
    )

    if not query:
        return {
            "query_type": "general"
        }

    # --------------------------------------------------------
    # Handle obvious small talk without calling the LLM.
    # --------------------------------------------------------

    if SMALLTALK_RE.match(query):

        return {
            "query_type": "general"
        }

    # --------------------------------------------------------
    # Classification prompt
    # --------------------------------------------------------

    prompt = f"""
You are a classifier for a college student assistant.

Student programme:
{state["programme"]}

Student question:
{query}

Classify the question into EXACTLY ONE category:

academic
fee
both
general

Definitions:

academic:
Questions about:
- attendance
- exams
- subjects
- courses
- marks
- grading
- academic rules
- academic policies
- semester information
- eligibility
- classes
- college academic requirements

fee:
Questions about:
- tuition fees
- admission fees
- semester fees
- examination fees
- hostel fees
- transport fees
- payment schedules
- fee structure
- refunds
- charges
- financial amounts

both:
Use this when the question clearly requires BOTH
academic information and fee information.

general:
Greetings, casual conversation, or questions that
cannot be answered using the academic handbook or fee structure.

Return ONLY one word:
academic
fee
both
general
"""

    response = classifier_llm.invoke(prompt)

    category = str(response.content).strip().lower()

    valid_categories = {
        "academic",
        "fee",
        "both",
        "general",
    }

    if category not in valid_categories:

        # Safer fallback for college-related questions.
        category = "academic"

    return {
        "query_type": category
    }


# ============================================================
# 13. ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State) -> dict:

    query = get_latest_user_message(
        state["messages"]
    )

    documents = academic_retriever.invoke(
        query
    )

    context = format_documents(
        documents
    )

    return {
        "retrieved_context": [
            f"ACADEMIC INFORMATION:\n{context}"
        ]
    }


# ============================================================
# 14. FEE RAG NODE
# ============================================================

def fee_rag_node(state: State) -> dict:

    query = get_latest_user_message(
        state["messages"]
    )

    documents = fee_retriever.invoke(
        query
    )

    context = format_documents(
        documents
    )

    return {
        "retrieved_context": [
            f"FEE INFORMATION:\n{context}"
        ]
    }


# ============================================================
# 15. GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:

    return {
        "retrieved_context": [
            "No college document retrieval is required for this question."
        ]
    }


# ============================================================
# 16. ROUTER
# ============================================================

def route_query(state: State):

    query_type = state["query_type"]

    routes = {
        "academic": ["academic_rag"],
        "fee": ["fee_rag"],
        "both": [
            "academic_rag",
            "fee_rag",
        ],
        "general": ["general"],
    }

    return routes.get(
        query_type,
        ["academic_rag"],
    )


# ============================================================
# 17. RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:

    query = get_latest_user_message(
        state["messages"]
    )

    programme = state["programme"]

    context = "\n\n".join(
        state.get(
            "retrieved_context",
            [],
        )
    )

    # --------------------------------------------------------
    # General conversation
    # --------------------------------------------------------

    if state["query_type"] == "general":

        prompt = f"""
You are a friendly college AI assistant.

Student programme:
{programme}

Student question:
{query}

Respond naturally and briefly.

If the student asks about college-specific information
that requires official documents, tell them you can help
with academic and fee information from the college documents.
"""

    # --------------------------------------------------------
    # RAG response
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a college AI assistant.

Student programme:
{programme}

Student question:
{query}

Retrieved official college information:
{context}

Instructions:

1. Answer the student's question using the retrieved information.
2. Do not invent facts.
3. Do not make up fees, dates, policies, or requirements.
4. If the information is not present in the retrieved context,
   clearly say that the provided college documents do not contain
   enough information to answer.
5. If the question is about fees, preserve exact amounts
   and payment information from the source.
6. If the question is about academic rules, preserve the
   actual requirements from the source.
7. If multiple pieces of information were retrieved, combine
   them into one clear answer.
8. Keep the response easy for a student to understand.
9. Do not mention internal implementation details such as
   LangGraph, vector databases, embeddings, or RAG.
10. Do not claim to have information that is not in the documents.

Answer:
"""

    response = response_llm.invoke(
        prompt
    )

    return {
        "messages": [response]
    }


# ============================================================
# 18. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(State)

# Nodes
builder.add_node(
    "rewrite",
    rewrite_node,
)

builder.add_node(
    "classifier",
    classifier_node,
)

builder.add_node(
    "academic_rag",
    academic_rag_node,
)

builder.add_node(
    "fee_rag",
    fee_rag_node,
)

builder.add_node(
    "general",
    general_node,
)

builder.add_node(
    "response",
    response_node,
)


# ============================================================
# 19. GRAPH EDGES
# ============================================================

builder.add_edge(
    START,
    "rewrite",
)

builder.add_edge(
    "rewrite",
    "classifier",
)

builder.add_conditional_edges(
    "classifier",
    route_query,
    [
        "academic_rag",
        "fee_rag",
        "general",
    ],
)

builder.add_edge(
    "academic_rag",
    "response",
)

builder.add_edge(
    "fee_rag",
    "response",
)

builder.add_edge(
    "general",
    "response",
)

builder.add_edge(
    "response",
    END,
)


# ============================================================
# 20. COMPILE GRAPH
# ============================================================

graph = builder.compile()


# ============================================================
# 21. STREAMLIT SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "programme" not in st.session_state:
    st.session_state.programme = None


# ============================================================
# 22. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 College Assistant")

    st.markdown(
        "Ask questions about your college's "
        "academic rules and fee structure."
    )

    st.divider()

    programme = st.selectbox(
        "Select your programme",
        [
            "BCA",
            "BBA",
            "B.com (H)",
        ],
        index=None,
        placeholder="Choose your programme",
    )

    if programme:

        st.session_state.programme = programme

    st.divider()

    st.subheader("Supported Questions")

    st.markdown(
        """
        **Academic**
        - Attendance
        - Exams
        - Subjects
        - Marks
        - Academic rules

        **Fees**
        - Tuition fees
        - Semester fees
        - Admission fees
        - Payment schedules

        **Combined**
        - Academic + fee questions
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 23. MAIN UI
# ============================================================

st.title("🎓 College AI Assistant")

st.caption(
    "Ask questions about academic policies and fee structure."
)


# ============================================================
# 24. REQUIRE PROGRAMME
# ============================================================

if not st.session_state.programme:

    st.info(
        "👈 Please select your programme from the sidebar first."
    )

    st.stop()


st.success(
    f"Programme selected: {st.session_state.programme}"
)


# ============================================================
# 25. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.write(
                message.content
            )

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.write(
                message.content
            )


# ============================================================
# 26. CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask your question..."
)


if user_input:

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(user_input)

    # --------------------------------------------------------
    # Add user message to Streamlit history
    # --------------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=user_input
        )
    )

    # --------------------------------------------------------
    # Build LangGraph input
    # --------------------------------------------------------

    graph_input: State = {
        "programme": st.session_state.programme,
        "messages": [
            HumanMessage(
                content=user_input
            )
        ],
        "query_type": "",
        "retrieved_context": [],
    }

    # --------------------------------------------------------
    # Run graph
    # --------------------------------------------------------

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                result = graph.invoke(
                    graph_input
                )

            # ------------------------------------------------
            # Get final AI message
            # ------------------------------------------------

            final_message = None

            for message in reversed(
                result["messages"]
            ):

                if isinstance(
                    message,
                    AIMessage,
                ):

                    final_message = message

                    break

            if final_message is None:

                final_answer = (
                    "Sorry, I could not generate an answer."
                )

            else:

                final_answer = str(
                    final_message.content
                )

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.write(
                final_answer
            )

            # ------------------------------------------------
            # Show classification
            # ------------------------------------------------

            query_type = result.get(
                "query_type",
                "general",
            )

            category_labels = {
                "academic": "📚 Academic",
                "fee": "💰 Fee",
                "both": "📚💰 Academic + Fee",
                "general": "💬 General",
            }

            st.caption(
                f"Category: "
                f"{category_labels.get(query_type, query_type)}"
            )

    except Exception as exc:

        final_answer = (
            "Sorry, something went wrong while processing "
            "your question."
        )

        st.error(
            final_answer
        )

        st.exception(exc)

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        AIMessage(
            content=final_answer
        )
    )