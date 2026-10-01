import operator
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Programme = Literal["BCA", "BBA", "B.com (H)"]
QueryType = Literal["academic", "fee", "both", "general"]


class State(TypedDict, total=False):
    programme: Programme
    messages: Annotated[list[BaseMessage], add_messages]
    rewritten_query: str
    query_type: QueryType
    # Each item is already source-labelled, e.g. "[Fee Structure, p.3] ..."
    retrieved_context: Annotated[list[str], operator.add]
    grounded: bool
    escalated: bool