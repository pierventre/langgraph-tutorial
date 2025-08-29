from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )

class Settings(BaseSettings):
    google_api_key: str

    class Config:
        env_file = ".env"

# Define a tool
@tool(description="Multiplies two numbers")
def multiply(a: int, b: int) -> int:
    return a * b

settings = Settings()

llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    google_api_key=settings.google_api_key
)

structured_llm = llm.with_structured_output(SearchQuery)
# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(output)

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

messages = [HumanMessage("What is 2 times 3?")]

# Invoke the LLM with input that triggers the tool call
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# Pass tool results back to the model
tool_message = ToolMessage(
    content=multiply(ai_message.tool_calls[0]['args']), 
    tool_call_id=ai_message.tool_calls[0]['id']
)
messages.append(tool_message)

final_response = llm_with_tools.invoke(messages)
print(final_response.content)
