# import os
# from typing import TypeGuard, TypedDict
# from unittest import result

# from requests import Response

# # Lets create the State First 

# class pipelinestate(TypedDict):
#     raw_input : str
#     edited_text : str
#     script_text : str
#     final_output : str
    

# from langchain_core.prompts import prompt
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv()

# llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.7)


# #1) ---------Editor Node 
# def editor_node(state: pipelinestate) -> dict:
#     """Stage 1 of the pipeline.
#     Cleans up grammar, removes typos, and refines the overall tone
#     of the provided content.
#     """
#     prompt = (
#         "You are an expert, copyeditor. Clean up the following raw text."
#         "Fix any grammatical errors, spelling mistakes, and smooth out the transitic"
#         "while keeping the core message intact. Return only the edited text.\n\n"
#         f"Text:\n{state ['raw_input']}"
#     )
#     response = llm.invoke(prompt)
#     return {"edited_text" : response.content}



# # 2)---------Scriptwriter Node 
# def scriptwriter_node(state: pipelinestate) -> dict:
#     """Stage 2 Format the clean text into a engaging video script style."""
#     print(f"\n---[Stage 2 Executing Scriptwriter Node---]")

#     prompt = (
#         "You are a charismatic YouTube content creator. Take this edited text and transform"
#         "it into a highly engaging, punchy, conversational video script hook. Make it sound"
#         "like a real person speaking passionately. Return only the script content.\n\n"
#         f"Edited Text:\n{state['edited_text']}"
#     )
#     response = llm.invoke(prompt)
#     return {"script_text" : response.content}



# # 3)---------Translater Node 
# def translator_node(state: pipelinestate) -> dict:
#     """Stage 3: Translate the script into natural flowing Hinglish"""
#     print(f"\n---[Stage 3 Executing Hinglish Translator Node---]")

#     prompt = (
#         "You are an expert content localizer for the Pakistan market. Take the following script"
#         "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence"
#         "or repeat information. Alternating comfortably between Hinglish and English phrases just like Humans"
#         "an intellectual tech educator would speak naturally on a live stream."
#         "Return only the final Hinglish text.\n\n"
#         f"Script:\n{state['script_text']}"
#     )
#     response = llm.invoke(prompt)
#     return {"final_output" : response.content}

# # Now your State and Nodes are ready and now it is time to create Graph
# # And for Creating the Graph you have to connect these Nodes and for that you have to use Edges
# # Edges are very important to create the Workflows

# from langgraph.graph import StateGraph, START, END

# # ---------Create the Graph 
# graph = StateGraph(pipelinestate)


# # ---------Add the Nodes in my Graph
# graph.add_node("editor",editor_node)
# graph.add_node("scriptwriter", scriptwriter_node)
# graph.add_node("translator", translator_node)


# # ---------Add Edges (sequential - One after another)
# graph.add_edge(START,"editor") 
# graph.add_edge("editor", "scriptwriter")
# graph.add_edge("scriptwriter", "translator")
# graph.add_edge("translator",END)


# # ---------Compile the Graph 
# app = graph.compile()

# result = app.invoke({
#     "raw_input" : "AI Agents are the Future of Tech. They can think, plan, and act on their own. LangGraph helps you build these Agents with proper control and memory"
# })

# # ---------Final Output
# print("Your result are : -\n\n") 
# print(result['final_output'])



# ---------------------------------------------------------------------------------------------------------

import os
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# ---------- State ----------
class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


llm = ChatGroq(model="meta-llama/llama-prompt-guard-2-22m", temperature=0.7)


# ---------- 1) Editor Node ----------
def editor_node(state: PipelineState) -> dict:
    """Stage 1: Clean up grammar, typos, and tone of the raw input."""
    print("\n---[Stage 1 Executing Editor Node]---")

    prompt = f"""You are an expert copyeditor. Clean up the following raw text.
Fix any grammatical errors, spelling mistakes, and smooth out the transitions
while keeping the core message intact. Return only the edited text.

Text:
{state['raw_input']}"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"[editor_node] LLM call failed: {e}")
        raise

    return {"edited_text": response.content}


# ---------- 2) Scriptwriter Node ----------
def scriptwriter_node(state: PipelineState) -> dict:
    """Stage 2: Format the clean text into an engaging video script style."""
    print("\n---[Stage 2 Executing Scriptwriter Node]---")

    prompt = f"""You are a charismatic YouTube content creator. Take this edited text and
transform it into a highly engaging, punchy, conversational video script hook.
Make it sound like a real person speaking passionately. Return only the script content.

Edited Text:
{state['edited_text']}"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"[scriptwriter_node] LLM call failed: {e}")
        raise

    return {"script_text": response.content}


# ---------- 3) Translator Node ----------
def translator_node(state: PipelineState) -> dict:
    """Stage 3: Translate the script into natural, flowing Hinglish."""
    print("\n---[Stage 3 Executing Hinglish Translator Node]---")

    prompt = f"""You are an expert content localizer for the Pakistan market. Take the
following script and convert it into natural, flowing 'Hinglish'. Do not simply
translate it sentence-by-sentence or repeat information. Alternate comfortably
between Hinglish and English phrases just like a real, intellectual tech
educator would speak naturally on a live stream. Return only the final Hinglish text.

Script:
{state['script_text']}"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        print(f"[translator_node] LLM call failed: {e}")
        raise

    return {"final_output": response.content}


# ---------- Build the Graph ----------
graph = StateGraph(PipelineState)

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

app = graph.compile()

# ---------- Run ----------
if __name__ == "__main__":
    result = app.invoke({
        "raw_input": (
            "AI Agents are the Future of Tech. They can think, plan, and act on "
            "their own. LangGraph helps you build these Agents with proper "
            "control and memory."
        )
    })

    print("\nFinal Result:\n")
    print(result["final_output"])


    