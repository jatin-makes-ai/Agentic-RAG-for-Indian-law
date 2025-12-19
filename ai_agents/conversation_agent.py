# Agents/conversation_agent.py
"""
Conversation agent for answering questions using retrieved chunks.
"""
from agents import Agent
from dotenv import load_dotenv
import os
load_dotenv()

prompt = """
You are a legal assistant. You will be provided a scenario as well as some relevant articles from the constitution of India that may be applicable in that scenario.
Your job is to answer the user's query/scenario based on the interpretation of the exact articles, citing the Articles wherever you use them.
"""

conversation_agent = Agent(
    name="Legal Assistant",
    model=os.getenv("CONVERSATION_LLM_MODEL", "gpt-4o-mini"),
    instructions=prompt,
)
