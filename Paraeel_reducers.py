from audioop import add
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph import graph
from langgraph.graph import StateGraph, START, END
from urllib3 import response

load_dotenv()

llm = ChatGroq(model="meta-llama/llama-prompt-guard-2-22m", temperature=0.1)


# ---------- Create a State ----------

def merge_score_dicts(existing : dict, newupdate : dict) -> dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}


class AnalyzaState(TypedDict):
    raw_text : str
    safety_score : Annotated[dict[str, int], merge_score_dicts]


# ---------- Create Nodes ----------

def toxicity_node(state : AnalyzaState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
    "Analyze the following text for profanity, aggression, hate speech, or toxicity."
    "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic"
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state ['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under single state key 
    return {"safety_score" : {"toxicity_score" : score}}

def copyright_node(state : AnalyzaState) -> dict:
    print("\n [Branch 2] Analyzing Copyright and Originality Risks...")
    prompt = (
    "Analyze the following text. Judge if it it sound heavily plagiarized, unorignal"
    "or present a coporate trademark risks.Provide a score from 0 to 100"
    "where 0 means entirely orignal and 100 means high risk"
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state ['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under single state key 
    return {"safety_score" : {"copyright_risk" : score}}


def cultural_node(state : AnalyzaState) -> dict:
    print("\n [Branch 3] Analyzing Regional and Cultural Sensitivity...")
    prompt = (
    "Analyze the following text and regional sensitivities, political landmines,"
    "or cultural insensitivity that might offend a globalaudience.Provide a score from 0 to 100"
    "where 0 means completely safe and 100 means highly offensive"
    "Return ONLY the plain integer number, nothing else.\n\n"
    f"Text:\n{state ['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return a sub-dictionary under single state key 
    return {"safety_score" : {"cultural_insensitivity" : score}}


builder = StateGraph(AnalyzaState)

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("cultural_node", cultural_node)

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "cultural_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("cultural_node", END)

app = builder.compile()

sample_script = """
Yo guys! Welcome back to the stream. Today I am going to show you how to hack into
your friend's system using a script I copied directly from an online forum.
Honestly, traditional security protocols are absolute garbage and anyone still using
them is an absolute idiot. Let's dive into the code!
"""

initial_state = {
    "raw_text" : sample_script,
    "safety_score" : {} 
}

final_state = app.invoke(initial_state)

print(final_state["safety_score"])



