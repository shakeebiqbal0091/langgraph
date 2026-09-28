# import os
# import streamlit as st
# from typing import TypedDict, Annotated
# from langgraph.graph.message import add_messages
# from langgraph.graph import StateGraph, START, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()

# # ----------------------------
# # Page config
# # ----------------------------
# st.set_page_config(
#     page_title="College Assistant",
#     page_icon="🎓",
#     layout="centered",
# )

# # ----------------------------
# # Step 1 - Building the RAG retrievers (cached so it runs only once)
# # ----------------------------
# @st.cache_resource(show_spinner="Loading knowledge base...")
# def load_resources():
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     def build_retriver(pdf_path: str):
#         loader = PyPDFLoader(pdf_path)
#         document = loader.load()

#         splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#         chunks = splitter.split_documents(document)
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         return vectorstore.as_retriever(search_kwargs={"k": 4})

#     acedemic_retriever = build_retriver("academics_handbook.pdf")
#     fee_retriever = build_retriver("fee_structure.pdf")

#     llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

#     return acedemic_retriever, fee_retriever, llm


# acedemic_retriever, fee_retriever, llm = load_resources()


# # ----------------------------
# # Step 2 - State
# # ----------------------------
# class State(TypedDict):
#     programme: str
#     messages: Annotated[list, add_messages]
#     query_type: str
#     retrieved_context: str


# # ----------------------------
# # Step 3 - Nodes generation
# # ----------------------------
# def classifier_node(state: State) -> dict:
#     """Look at the latest user message and decide which path to take."""

#     last_message = state['messages'][-1].content

#     prompt = (
#         "Classify the following student query into exactly one category: "
#         "'academic', 'fee', or 'general'.\n\n"
#         "Use 'academic' for questions about attendance, exams, grading, credits, "
#         "promotion, course structure, summer training, or degree requirements.\n"
#         "Use 'fee' for questions about tuition, payment, refund, late charges, "
#         "scholarships, or any money-related topic.\n"
#         "Use 'general' for greetings, casual talk, or anything not related to "
#         "the college rules or fee.\n\n"
#         f"Query: {last_message}\n\n"
#         "Return only one word: academic, fee, or general."
#     )

#     response = llm.invoke(prompt)
#     category = response.content.strip().lower()

#     if "academic" in category:
#         category = "academic"
#     elif "fee" in category:
#         category = "fee"
#     else:
#         category = "general"

#     return {"query_type": category}


# def academic_rag_node(state: State) -> dict:
#     """Retrieves relevant chunks from the academics handbook."""
#     query = state["messages"][-1].content
#     docs = acedemic_retriever.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     return {"retrieved_context": context}


# def fee_rag_node(state: State) -> dict:
#     """Retrieves relevant chunks from the fee structure PDF."""
#     query = state["messages"][-1].content
#     docs = fee_retriever.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     return {"retrieved_context": context}


# def general_node(state: State) -> dict:
#     """Answers directly using the LLM's own knowledge, no retrieval needed."""
#     return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


# def response_node(state: State) -> dict:
#     """Generates the final answer, personalized using the student's programme."""
#     query = state["messages"][-1].content
#     programme = state.get("programme", "Unknown")
#     context = state["retrieved_context"]

#     if context == "NO_RETRIEVAL_NEEDED":
#         prompt = (
#             f"You are a friendly college assistant talking to a {programme} student. "
#             f"Answer this question using your own general knowledge:\n\n{query}"
#         )
#     else:
#         prompt = (
#             f"You are a college assistant helping a {programme} student. "
#             f"Use the following context from the official college documents to answer "
#             f"the question accurately. If the context mentions specific figures for "
#             f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question: {query}\n\n"
#             f"Give a clear, friendly, and precise answer."
#         )

#     response = llm.invoke(prompt)
#     return {"messages": [("ai", response.content.strip())]}


# # ----------------------------
# # Step 4 - router function
# # ----------------------------
# def route_query(state: State):
#     if state['query_type'] == 'academic':
#         return "academic_rag"
#     elif state['query_type'] == "fee":
#         return "fee_rag"
#     else:
#         return "general"


# # ----------------------------
# # Step 5 - Building the graph (cached)
# # ----------------------------
# @st.cache_resource(show_spinner=False)
# def build_graph():
#     graph = StateGraph(State)

#     graph.add_node("classifier", classifier_node)
#     graph.add_node("academic_rag", academic_rag_node)
#     graph.add_node("fee_rag", fee_rag_node)
#     graph.add_node("general", general_node)
#     graph.add_node("response", response_node)

#     graph.add_edge(START, "classifier")

#     graph.add_conditional_edges("classifier", route_query)

#     graph.add_edge("academic_rag", "response")
#     graph.add_edge("fee_rag", "response")
#     graph.add_edge("general", "response")

#     graph.add_edge("response", END)

#     return graph.compile()


# app = build_graph()

# # ----------------------------
# # Step 6 - Streamlit UI
# # ----------------------------

# # --- Custom CSS ---
# st.markdown("""
# <style>
# .main-header {
#     text-align: center;
#     padding: 1rem 0 0.5rem 0;
# }
# .main-header h1 {
#     font-size: 2.2rem;
#     margin-bottom: 0.2rem;
# }
# .main-header p {
#     color: #888;
#     font-size: 0.95rem;
# }
# .stChatMessage {
#     border-radius: 12px;
# }
# div[data-testid="stChatInput"] {
#     border-radius: 12px;
# }
# .query-badge {
#     display: inline-block;
#     padding: 2px 10px;
#     border-radius: 999px;
#     font-size: 0.7rem;
#     font-weight: 600;
#     margin-bottom: 4px;
# }
# .badge-academic { background-color: #1f3a5f; color: #93c5fd; }
# .badge-fee { background-color: #4a3110; color: #fcd34d; }
# .badge-general { background-color: #1f4a2e; color: #86efac; }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="main-header">
#     <h1>🎓 College Assistant</h1>
#     <p>Ask me about academics, fees, or anything else campus-related</p>
# </div>
# """, unsafe_allow_html=True)

# # --- Sidebar: programme selection ---
# with st.sidebar:
#     st.header("⚙️ Setup")

#     programme_map = {
#         "BCA": "BCA",
#         "BBA": "BBA",
#         "B.Com (H)": "B.Com (H)",
#     }

#     student_programme = st.selectbox(
#         "Select your programme",
#         options=list(programme_map.keys()),
#         index=0,
#     )

#     st.markdown("---")
#     st.caption(f"📌 Currently set as: **{student_programme}** student")

#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.messages = []
#         st.session_state.lc_messages = []
#         st.rerun()

#     st.markdown("---")
#     st.caption("Routes queries to:")
#     st.caption("📘 Academic Handbook (RAG)")
#     st.caption("💰 Fee Structure (RAG)")
#     st.caption("💬 General Knowledge")

# # --- Session state init ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []  # for display: list of {"role", "content", "query_type"}

# if "lc_messages" not in st.session_state:
#     st.session_state.lc_messages = []  # actual langgraph message history (human/ai tuples)

# # --- Render chat history ---
# for msg in st.session_state.messages:
#     avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
#     with st.chat_message(msg["role"], avatar=avatar):
#         if msg["role"] == "assistant" and msg.get("query_type"):
#             badge_class = f"badge-{msg['query_type']}"
#             st.markdown(
#                 f'<span class="query-badge {badge_class}">{msg["query_type"].upper()}</span>',
#                 unsafe_allow_html=True
#             )
#         st.markdown(msg["content"])

# # --- Chat input ---
# user_query = st.chat_input("Type your question here...")

# if user_query:
#     # Show user message
#     st.session_state.messages.append({"role": "user", "content": user_query})
#     with st.chat_message("user", avatar="🧑‍🎓"):
#         st.markdown(user_query)

#     # Append to LangGraph message history
#     st.session_state.lc_messages.append(("human", user_query))

#     # Invoke the graph
#     with st.chat_message("assistant", avatar="🎓"):
#         with st.spinner("Thinking..."):
#             result = app.invoke({
#                 "programme": student_programme,
#                 "messages": st.session_state.lc_messages
#             })

#             ai_response = result["messages"][-1].content
#             query_type = result.get("query_type", "general")

#             badge_class = f"badge-{query_type}"
#             st.markdown(
#                 f'<span class="query-badge {badge_class}">{query_type.upper()}</span>',
#                 unsafe_allow_html=True
#             )
#             st.markdown(ai_response)

#     # Update histories
#     st.session_state.lc_messages = result["messages"]
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": ai_response,
#         "query_type": query_type
#     })



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

    messages: Annotated[
        list,
        add_messages
    ]

    query_type: str

    retrieved_context: str


# ============================================================
# Step 3 - Classifier Node
# ============================================================

def classifier_node(state: State) -> dict:

    """
    Looks at the latest user message and classifies it into:

    academic
    fee
    general
    """

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly "
        "one category: academic, fee, or general.\n\n"

        "Use 'academic' for questions about:\n"
        "- attendance\n"
        "- exams\n"
        "- grading\n"
        "- credits\n"
        "- promotion\n"
        "- course structure\n"
        "- summer training\n"
        "- degree requirements\n"
        "- academic rules\n\n"

        "Use 'fee' for questions about:\n"
        "- tuition\n"
        "- payment\n"
        "- refund\n"
        "- late charges\n"
        "- scholarships\n"
        "- fees\n"
        "- any money-related college topic\n\n"

        "Use 'general' for:\n"
        "- greetings\n"
        "- casual conversation\n"
        "- general questions\n"
        "- anything unrelated to academic rules or fees\n\n"

        f"Student query:\n{last_message}\n\n"

        "Return ONLY ONE WORD:\n"
        "academic\n"
        "fee\n"
        "general"
    )

    response = classifier_llm.invoke(prompt)

    category = str(response.content).strip().lower()

    # --------------------------------------------------------
    # Normalize classifier output
    # --------------------------------------------------------

    if "academic" in category:

        category = "academic"

    elif "fee" in category:

        category = "fee"

    else:

        category = "general"

    return {
        "query_type": category
    }


# ============================================================
# Step 4 - Academic RAG Node
# ============================================================

def academic_rag_node(state: State) -> dict:

    """
    Retrieves relevant information from
    the academic handbook.
    """

    query = state["messages"][-1].content

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

    query = state["messages"][-1].content

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

    query = state["messages"][-1].content

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

        prompt = (
            "You are a friendly college assistant.\n\n"

            f"The student is enrolled in the "
            f"{programme} programme.\n\n"

            "Answer the student's question naturally "
            "and conversationally using your general knowledge.\n\n"

            "Do not mention the classification system.\n"
            "Do not mention RAG.\n"
            "Do not mention internal tools.\n\n"

            f"Student question:\n{query}\n\n"

            "Give a helpful and concise answer."
        )

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

    graph.add_edge(
        START,
        "classifier"
    )

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