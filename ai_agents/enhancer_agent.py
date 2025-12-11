from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import os
load_dotenv()

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


prompt = """
You are an assistant to a famous professional legal advisor.
Your task is to take the user's query and enhance it to make it easier to be looked up in the Indian Constitution.
You must only return the enhanced query, no other text or explanation.
"""

enhancer_agent = Agent(
    name="Query Enhancer",
    model=os.getenv("CONVERSATION_LLM_MODEL", "gpt-4o-mini"),
    instructions=prompt,
)