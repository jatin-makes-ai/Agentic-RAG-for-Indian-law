# Agents/conversation_agent.py
"""
Conversation agent for answering questions using retrieved chunks.
"""
from agents import Agent
from dotenv import load_dotenv
import os
load_dotenv()

prompt = """
You are a legal assistant. Answer concisely and cite the chunk_ids used.
If the answer cannot be found in the provided context, say "I don't know".
Use only the retrieved chunks provided to you to answer the question.
Cite chunk ids in your answer when you reference them.
"""

conversation_agent = Agent(
    name="Legal Assistant",
    model=os.getenv("CONVERSATION_LLM_MODEL", "gpt-4o-mini"),
    instructions=prompt,
)
