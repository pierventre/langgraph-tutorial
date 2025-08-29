from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Literal
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

class Settings(BaseSettings):
    google_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()

# Graph state
class State(TypedDict):
    joke: str
    improved_joke: str
    topic: str
    feedback: str
    funny_or_not: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    google_api_key=settings.google_api_key
)

# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""

    ret = {}
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
        ret["improved_joke"] = msg.content
    else:
        msg = llm.invoke(f"Write a sad story about {state['topic']}")
        ret["joke"] = msg.content
    return ret


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    if state.get("improved_joke"):
        grade = evaluator.invoke(f"Grade the joke {state['improved_joke']}")
    else:
        grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

# Build workflow
evaluator_builder = StateGraph(State)

# Add the nodes
evaluator_builder.add_node("llm_call_generator", llm_call_generator)
evaluator_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
evaluator_builder.add_edge(START, "llm_call_generator")
evaluator_builder.add_edge("llm_call_generator", "llm_call_evaluator")
evaluator_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
evaluator_workflow = evaluator_builder.compile()

# Show the workflow
img_bytes = evaluator_workflow.get_graph().draw_mermaid_png()
with open("evaluator_workflow.png", "wb") as f:
    f.write(img_bytes)

# Invoke
state = evaluator_workflow.invoke({"topic": "Cats"})
print(state["joke"] + "\n----------")
print(state["feedback"] + "\n----------")
print(state["improved_joke"] + "\n----------")
