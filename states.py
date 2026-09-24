# So now we are creating a Graph 
# And the first thing you create is a State 


import os


# 1)- Types Dict (Most Common Approach)

from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int

# 2)- Pydantic Approach
# It is good at data validation and type checking at 
# Runtime 

from pydantic import BaseModel, field_validator

class State1(BaseModel):
    topic : str
    summary : str = ""
    score : int

    @field_validator("score")
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be Positive")


# 3)- Python dataclasses
# Standard Python classes but it is used very rarely 

from dataclasses import dataclass, field
from langgraph.graph import MessagesState

@dataclass
class State2:
    topic : str = ""
    summary : str = ""
    message : list = field(default_factory=list)


# 4)- langgraph Message State


class State3(MessagesState):
    # Message field is already included with add_message reducer 
    # Just add your extra field 

    user_name : str
    language : str
    