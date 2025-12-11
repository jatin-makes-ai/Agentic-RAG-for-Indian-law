# ai_agents/retrieve_checker_agent.py
"""
Retrieve checker agent that reviews retrieved chunks and provides feedback.
"""
from agents import Agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
load_dotenv()

class RetrieveCheckerResponse(BaseModel):
    sufficiency: bool = Field(description="Is the context in the retrieved chunks collectively sufficient to answer the question?")
    correctness: bool = Field(description="Are the retrieved chunks collectively correct and relevant to the user query?")
    feedback: str = Field(description="Specific feedback to help improve the query for better retrieval (if sufficiency or correctness is false, provide actionable feedback)")

prompt = """
You are a retrieval quality checker for a legal document search system.
Your task is to review retrieved chunks and evaluate them against a user query.

You must output your evaluation as a JSON object with exactly these fields:
1. "sufficiency": boolean - Is the context in the retrieved chunks collectively sufficient to answer the question?
2. "correctness": boolean - Are the retrieved chunks collectively correct and relevant to the user query?
3. "feedback": string - Specific feedback to help improve the query for better retrieval (if sufficiency or correctness is false, provide actionable feedback)

Output ONLY valid JSON, no additional text or explanation.
Example format:
{
  "sufficiency": true,
  "correctness": true,
  "feedback": ""
}

Or if issues found:
{
  "sufficiency": false,
  "correctness": true,
  "feedback": "The retrieved chunks lack information about Article 21. Consider adding 'fundamental rights' or 'Article 21' to the query."
}
"""

retrieve_checker_agent = Agent(
    name="Retrieve Checker",
    model=os.getenv("RETRIEVE_CHECKER_AGENT_MODEL", "gpt-4o-mini"),
    instructions=prompt,
    output_type=RetrieveCheckerResponse
)

