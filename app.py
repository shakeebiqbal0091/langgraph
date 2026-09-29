import streamlit as st
import pdfplumber
from langchain_core.documents import Document

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

import re
from langchain_core.messages import AIMessage
import operator   # add to imports

SMALLTALK_RE = re.compile(
    r"^\s*("
    r"hi+|hello+|hey+|salam|assalam[\w\s-]*|"
    r"good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you|"
    r"thanks?(\s+you)?|thank\s+u|thx|"
    r"ok(ay)?|cool|great|"
    r"bye|goodbye|see\s+you"
    r")[\s!.?,]*$",
    re.IGNORECASE,
)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }

    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }

    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    div[data-testid="stChatInput"] {
        border-radius: 12px;
    }

    .query-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .badge-academic {
        background-color: #1f3a5f;
        color: #93c5fd;
    }

    .badge-fee {
        background-color: #4a3110;
        color: #fcd34d;
    }

    .badge-general {
        background-color: #1f4a2e;
        color: #86efac;
    }
    .badge-both {
    background-color: #3b2a5f;
    color: #d8b4fe;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 College Assistant</h1>
        <p>
            Ask me about academics, fees, exams,
            attendance, or general questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Step 1 - Load RAG Resources
# ============================================================

PROGRAMMES = ["BCA", "BBA", "B.Com (H)"]


def load_fee_documents(pdf_path: str) -> list[Document]:
    """
    Fee PDF -> documents.
      - each table row  -> one document: "Header: value | Header: value"
      - everything else -> normal 800-char text chunks
    """
    docs: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]

            # 1) Table rows
            for table in tables:
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue

                header = [
                    (h or "").replace("\n", " ").strip() or f"col{i}"
                    for i, h in enumerate(rows[0])
                ]
                last_first_cell = ""

                for row in rows[1:]:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if not any(cells):
                        continue

                    # Merged cells: forward-fill the first column
                    if cells[0]:
                        last_first_cell = cells[0]
                    else:
                        cells[0] = last_first_cell

                    text = " | ".join(
                        f"{h}: {c}" for h, c in zip(header, cells) if c
                    )
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path,
                                "page": page_no,
                                "kind": "table_row",
                            },
                        )
                    )

            # 2) Prose outside the tables
            def outside_tables(obj, bboxes=bboxes):
                return not any(
                    obj["x0"] >= b[0] and obj["x1"] <= b[2]
                    and obj["top"] >= b[1] and obj["bottom"] <= b[3]
                    for b in bboxes
                )

            prose = page.filter(outside_tables).extract_text() or ""
            for chunk in splitter.split_text(prose):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_path,
                            "page": page_no,
                            "kind": "text",
                        },
                    )
                )

    return docs


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Build Retriever Function
    # --------------------------------------------------------

    def build_retriever(pdf_path: str):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    # --------------------------------------------------------
    # Academic Retriever
    # --------------------------------------------------------

    academic_retriever = build_retriever(
        "academics_handbook.pdf"
    )

    # --------------------------------------------------------
    # Fee Retriever
    # --------------------------------------------------------

    fee_retriever = build_retriever(
        "fee_structure.pdf"
    )

    # --------------------------------------------------------
    # LLM for Classification
    # --------------------------------------------------------

    classifier_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # --------------------------------------------------------
    # LLM for Final Answers
    # --------------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.4,
    )

    return (
        academic_retriever,
        fee_retriever,
        classifier_llm,
        llm,
    )


(
    academic_retriever,
    fee_retriever,
    classifier_llm,
    llm,
) = load_resources()


# ============================================================
# Step 2 - LangGraph State
# ============================================================

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    standalone_query: str
    query_type: str
    retrieved_context: Annotated[list[str], operator.add]   # was: str

# ============================================================
# Step 2b - Query Rewrite Node (resolves follow-ups)
# ============================================================
def rewrite_node(state: State) -> dict:
    """
    Turns the latest message into a self-contained query using recent
    history, so routing and retrieval work on follow-ups like
    "and for BBA?". First turns skip the LLM call entirely.
    """
    messages = state["messages"]
    latest = messages[-1].content

    if len(messages) == 1:
        return {"standalone_query": latest}

    history = "\n".join(
        f"{'Student' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in messages[-7:-1]          # last 6 messages before the latest
    )

    prompt = (
        "Rewrite the student's latest message as a single self-contained "
        "question, using the conversation history to resolve references "
        "like 'it', 'that', 'and for BBA?', 'what about the late fee?'.\n\n"
        "Rules:\n"
        "- If the message is already self-contained, return it unchanged.\n"
        "- Do not answer the question.\n"
        "- Do not add facts or details that are not in the conversation.\n"
        "- Output ONLY the rewritten question, nothing else.\n\n"
        f"Conversation history:\n{history}\n\n"
        f"Latest message:\n{latest}\n\n"
        "Rewritten question:"
    )

    rewritten = str(classifier_llm.invoke(prompt).content).strip()
    return {"standalone_query": rewritten or latest}


# ============================================================
# Step 3 - Classifier Node
# ============================================================

def classifier_node(state: State) -> dict:
    query = state["standalone_query"]

    prompt = (
        "Classify the student's query into exactly one category: "
        "academic, fee, both, or general.\n\n"
        "academic: attendance, exams, grading, credits, promotion, course "
        "structure, summer training, degree requirements, academic rules, "
        "and ANY other question about how this college works (calendar, "
        "policies, facilities, staff, hostel, procedures).\n\n"
        "fee: tuition, payment, refund, late charges, scholarships, or any "
        "money-related college topic.\n\n"
        "both: the answer genuinely needs BOTH the academic rules AND the "
        "fee/money rules (e.g. a fee consequence of an academic event, or "
        "an academic consequence of a payment event). Do not use 'both' "
        "just because a query mentions two words; use it only when a "
        "correct answer needs facts from each document.\n\n"
        "general: ONLY greetings, thanks, small talk, or questions clearly "
        "unrelated to any college (e.g. 'what is Python?').\n\n"
        "IMPORTANT: If the query could plausibly be about this college and "
        "you are unsure, choose academic, fee, or both, NEVER general.\n\n"
        "Examples:\n"
        "'hi there' -> general\n"
        "'explain recursion' -> general\n"
        "'what is the minimum attendance?' -> academic\n"
        "'is there a hostel?' -> academic\n"
        "'what is the late payment charge?' -> fee\n"
        "'can I get a scholarship?' -> fee\n"
        "'do I get a refund if I fail attendance and get detained?' -> both\n"
        "'can I sit the exam if my fee is unpaid?' -> both\n"
        "'what fee do I pay to re-take a failed subject?' -> both\n\n"
        f"Query: {query}\n\n"
        "Return ONLY ONE WORD: academic, fee, both, or general."
    )

    response = classifier_llm.invoke(prompt)
    category = str(response.content).strip().lower()

    # Normalize. Order matters: check 'both' first.
    # Unrecognized output falls back to academic RAG, never general.
    if "both" in category:
        category = "both"
    elif "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    elif "general" in category:
        category = "general"
    else:
        category = "academic"

    return {"query_type": category}

# ============================================================
# Step 4 - Academic RAG Node
# ============================================================

def academic_rag_node(state: State) -> dict:
    documents = academic_retriever.invoke(state["standalone_query"])
    context = "\n\n".join(d.page_content for d in documents)
    return {"retrieved_context": [f"[ACADEMIC HANDBOOK]\n{context}"]}

# ============================================================
# Step 5 - Fee RAG Node
# ============================================================

def fee_rag_node(state: State) -> dict:
    """Retrieves fee rows/rules, biased toward the student's programme."""
    query = state["standalone_query"]
    programme = state.get("programme", "")

    # Row documents contain the programme name, so prefixing it improves
    # matching. Skip the prefix if the student named a programme themselves.
    names_a_programme = any(p.lower() in query.lower() for p in PROGRAMMES)
    search_query = query if names_a_programme else f"{programme} {query}".strip()

    documents = fee_retriever.invoke(search_query)
    context = "\n\n".join(d.page_content for d in documents)
    return {"retrieved_context": [f"[FEE STRUCTURE]\n{context}"]}

# ============================================================
# Step 6 - General Node
# ============================================================

def general_node(state: State) -> dict:
    return {"retrieved_context": []}


# ============================================================
# Step 7 - Response Node
# ============================================================

def response_node(state: State) -> dict:
    query = state["standalone_query"]
    programme = state.get("programme", "Unknown")

    # Parallel branches finish in nondeterministic order; sort for stable
    # prompts ("[ACADEMIC..." < "[FEE..." alphabetically).
    sections = sorted(state.get("retrieved_context") or [])

    # General path: greetings / non-college questions
    if not sections:
        prompt = (
            "You are a friendly college assistant chatting with a student "
            f"in the {programme} programme.\n\n"
            "This is a greeting, small talk, or a general non-college "
            "question. Answer naturally and concisely.\n\n"
            "STRICT RULES:\n"
            "- Do NOT state or guess anything about this college's "
            "policies, fees, dates, rules, staff, or facilities.\n"
            "- If the message actually asks about the college, say you can "
            "look that up in the official documents and ask them to "
            "rephrase their question about academics or fees.\n"
            "- Do not mention classification, RAG, or internal tools.\n\n"
            f"Student message:\n{query}\n\n"
            "Answer:"
        )
    else:
        context = "\n\n---\n\n".join(sections)
        prompt = (
            "You are a college assistant helping a student.\n\n"
            f"The student is enrolled in the {programme} programme.\n\n"
            "Use the following official college document excerpts to "
            "answer the question accurately. Each excerpt is labelled "
            "with its source document.\n\n"
            "IMPORTANT RULES:\n"
            "1. Use the provided excerpts as the only source of college "
            "facts.\n"
            "2. Do not invent college policies, dates, or numbers.\n"
            "3. If any part of the question is not covered by the "
            "excerpts, say clearly that this part was not found in the "
            "provided college documents. Answer the covered parts "
            "normally.\n"
            "4. If the answer uses more than one document, state which "
            "document each fact comes from.\n"
            f"5. If figures differ by programme, focus on {programme}.\n"
            "6. Give a clear and friendly answer.\n\n"
            f"Official document excerpts:\n{context}\n\n"
            f"Student question:\n{query}\n\n"
            "Answer:"
        )

    response = llm.invoke(prompt)
    return {"messages": [response]}

# ============================================================
# Step 8 - Router
# ============================================================

def route_query(state: State):
    """Returning a list of node names runs them in parallel."""
    return {
        "academic": ["academic_rag"],
        "fee": ["fee_rag"],
        "both": ["academic_rag", "fee_rag"],
    }.get(state["query_type"], ["general"])

# ============================================================
# Step 9 - Build LangGraph
# ============================================================

@st.cache_resource(show_spinner=False)
def build_graph():

    graph = StateGraph(State)

    # --------------------------------------------------------
    # Add Nodes
    # --------------------------------------------------------

    graph.add_node("rewrite", rewrite_node)       # NEW, next to the other add_node calls


    graph.add_node(
        "classifier",
        classifier_node
    )

    graph.add_node(
        "academic_rag",
        academic_rag_node
    )

    graph.add_node(
        "fee_rag",
        fee_rag_node
    )

    graph.add_node(
        "general",
        general_node
    )

    graph.add_node(
        "response",
        response_node
    )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    # replace: graph.add_edge(START, "classifier")
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "classifier")

    # --------------------------------------------------------
    # Conditional Routing
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classifier",
        route_query
    )

    # --------------------------------------------------------
    # RAG -> Response
    # --------------------------------------------------------

    graph.add_edge(
        "academic_rag",
        "response"
    )

    graph.add_edge(
        "fee_rag",
        "response"
    )

    graph.add_edge(
        "general",
        "response"
    )

    # --------------------------------------------------------
    # Response -> End
    # --------------------------------------------------------

    graph.add_edge(
        "response",
        END
    )

    return graph.compile()


app = build_graph()


# ============================================================
# Step 10 - Sidebar
# ============================================================

with st.sidebar:

    st.header("⚙️ Setup")

    programme_options = [
        "BCA",
        "BBA",
        "B.Com (H)",
    ]

    student_programme = st.selectbox(
        "Select your programme",
        options=programme_options,
        index=0,
    )

    st.markdown("---")

    st.caption(
        f"📌 Currently set as: **{student_programme}** student"
    )

    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.session_state.lc_messages = []

        st.rerun()

    st.markdown("---")

    st.caption("Routes queries to:")

    st.caption("📘 Academic Handbook (RAG)")

    st.caption("💰 Fee Structure (RAG)")

    st.caption("💬 General Knowledge")


# ============================================================
# Step 11 - Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "lc_messages" not in st.session_state:

    st.session_state.lc_messages = []


# ============================================================
# Step 12 - Display Previous Messages
# ============================================================

for msg in st.session_state.messages:

    avatar = (
        "🧑‍🎓"
        if msg["role"] == "user"
        else "🎓"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar,
    ):

        # ----------------------------------------------------
        # Query Type Badge
        # ----------------------------------------------------

        if (
            msg["role"] == "assistant"
            and msg.get("query_type")
        ):

            query_type = msg["query_type"]

            badge_class = f"badge-{query_type}"

            st.markdown(
                f"""
                <span class="query-badge {badge_class}">
                    {query_type.upper()}
                </span>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # Message
        # ----------------------------------------------------

        st.markdown(
            msg["content"]
        )


# ============================================================
# Step 13 - Chat Input
# ============================================================

user_query = st.chat_input(
    "Type your question here..."
)


# ============================================================
# Step 14 - Process User Query
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑‍🎓",
    ):

        st.markdown(user_query)

    # --------------------------------------------------------
    # Add Human Message to LangGraph History
    # --------------------------------------------------------

    st.session_state.lc_messages.append(
        (
            "human",
            user_query,
        )
    )

    # --------------------------------------------------------
    # Invoke LangGraph
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🎓",
    ):

        with st.spinner("Thinking..."):

            result = app.invoke(
                {
                    "programme": student_programme,
                    "messages": st.session_state.lc_messages,
                }
            )

            # ------------------------------------------------
            # Get Final AI Response
            # ------------------------------------------------

            ai_message = result["messages"][-1]

            ai_response = ai_message.content

            # ------------------------------------------------
            # Get Classification
            # ------------------------------------------------

            query_type = result.get(
                "query_type",
                "general",
            )

        # ----------------------------------------------------
        # Display Category Badge
        # ----------------------------------------------------

        badge_class = f"badge-{query_type}"

        st.markdown(
            f"""
            <span class="query-badge {badge_class}">
                {query_type.upper()}
            </span>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Display Final Answer
        # ----------------------------------------------------

        st.markdown(ai_response)

    # --------------------------------------------------------
    # Update LangGraph History
    # --------------------------------------------------------

    st.session_state.lc_messages = result["messages"]

    # --------------------------------------------------------
    # Update Streamlit History
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response,
            "query_type": query_type,
        }
    )