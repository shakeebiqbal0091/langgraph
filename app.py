import streamlit as st

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
    standalone_query: str          # NEW
    query_type: str
    retrieved_context: str

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
    """
    Step 1: deterministic smalltalk check (no LLM).
    Step 2: binary LLM decision, academic vs fee.
    There is deliberately no LLM-decided 'general' route.
    """
    query = state["standalone_query"]

    if SMALLTALK_RE.match(query):
        return {"query_type": "general"}

    prompt = (
        "Classify the student's query about their college into exactly one "
        "category: academic or fee.\n\n"
        "fee: tuition, payment, refund, late charges, scholarships, "
        "installments, or any money-related topic.\n"
        "academic: everything else (attendance, exams, grading, credits, "
        "promotion, courses, calendar, rules, facilities, staff, hostel, "
        "procedures).\n\n"
        "Examples:\n"
        "'what is the minimum attendance?' -> academic\n"
        "'who is the principal?' -> academic\n"
        "'is there a hostel?' -> academic\n"
        "'what is the late payment charge?' -> fee\n"
        "'can I pay in installments?' -> fee\n\n"
        f"Query: {query}\n\n"
        "Return ONLY ONE WORD: academic or fee."
    )

    text = str(classifier_llm.invoke(prompt).content).strip().lower()

    # Default is academic; anything unparseable still goes through RAG.
    category = "fee" if ("fee" in text and "academic" not in text) else "academic"

    return {"query_type": category}

# ============================================================
# Step 4 - Academic RAG Node
# ============================================================

def academic_rag_node(state: State) -> dict:

    """
    Retrieves relevant information from
    the academic handbook.
    """

    query = state["standalone_query"]             # was state["messages"][-1].content

    documents = academic_retriever.invoke(query)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# Step 5 - Fee RAG Node
# ============================================================

def fee_rag_node(state: State) -> dict:

    """
    Retrieves relevant information from
    the fee structure document.
    """

    query = state["standalone_query"]             # was state["messages"][-1].content

    documents = fee_retriever.invoke(query)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# Step 6 - General Node
# ============================================================

def general_node(state: State) -> dict:

    """
    General questions do not require PDF retrieval.
    """

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# Step 7 - Response Node
# ============================================================

def response_node(state: State) -> dict:

    """
    Generates the final answer using the retrieved
    context when required.
    """

    query = state["standalone_query"]             # was state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state.get(
        "retrieved_context",
        "NO_RETRIEVAL_NEEDED"
    )

    # --------------------------------------------------------
    # General Question
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":
        # Fixed template: no LLM call, so nothing can be hallucinated.
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Happy to help! I can answer questions about "
                        f"academics (attendance, exams, promotion, credits) "
                        f"and fees for {programme}. What would you like to know?"
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # RAG Question
    # --------------------------------------------------------

    else:

        prompt = (
            "You are a college assistant helping a student.\n\n"

            f"The student is enrolled in the "
            f"{programme} programme.\n\n"

            "Use the following official college document "
            "context to answer the question accurately.\n\n"

            "IMPORTANT RULES:\n"
            "1. Use the provided context as the primary source.\n"
            "2. Do not invent college policies or numbers.\n"
            "3. If the answer is not present in the context, "
            "clearly say that the information was not found "
            "in the provided college documents.\n"
            "4. If multiple programmes are mentioned, "
            f"focus on {programme} when possible.\n"
            "5. Give a clear and friendly answer.\n\n"

            f"Official document context:\n"
            f"{context}\n\n"

            f"Student question:\n"
            f"{query}\n\n"

            "Answer:"
        )

    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    # Return actual AIMessage
    return {
        "messages": [response]
    }


# ============================================================
# Step 8 - Router
# ============================================================

def route_query(state: State):

    """
    Routes the query according to the classifier result.
    """

    query_type = state["query_type"]

    if query_type == "academic":

        return "academic_rag"

    elif query_type == "fee":

        return "fee_rag"

    else:

        return "general"


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