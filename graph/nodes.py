from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

class Classification(BaseModel):
    query_type: QueryType = Field(description="academic | fee | both | general")

_classifier = ChatGroq(model="openai/gpt-oss-20b", temperature=0).with_structured_output(Classification)

CLASSIFIER_PROMPT = """Classify the student's question for a college assistant.
- academic: attendance, exams, marks, grading, subjects, semesters, eligibility, rules
- fee: tuition/admission/exam/hostel/transport fees, payment schedules, refunds
- both: clearly needs academic AND fee information
- general: greetings, small talk, unrelated to college documents
Programme: {programme}
Question: {question}"""

def classifier_node(state: State) -> dict:
    question = state.get("rewritten_query") or state["messages"][-1].content
    try:
        result = _classifier.invoke(
            CLASSIFIER_PROMPT.format(programme=state.get("programme", "unknown"), question=question)
        )
        return {"query_type": result.query_type}
    except Exception:
        # Fail safe: route to retrieval rather than answering from thin air
        return {"query_type": "both"}