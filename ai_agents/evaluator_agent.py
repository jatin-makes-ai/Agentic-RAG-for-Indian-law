# ai_agents/evaluator_agent.py
"""
Evaluator agent that reviews conversation agent responses and provides feedback.
"""
from agents import Agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
load_dotenv()

class EvaluatorResponse(BaseModel):
    is_acceptable: bool = Field(description="Is the response acceptable and fit for the user?")
    completeness: bool = Field(description="Is the response complete and addresses all aspects of the question?")
    accuracy: bool = Field(description="Is the response accurate based on the provided context?")
    proper_citation: bool = Field(description="Does the response properly cite chunk_ids when referencing information?")
    feedback: str = Field(description="Specific feedback to improve the response (if any flag is false, provide actionable feedback)")

prompt = """
You are a response quality evaluator for a legal document search system.
Your task is to review the conversation agent's response and evaluate its quality.

You must output your evaluation as a JSON object with exactly these fields:
1. "is_acceptable": boolean - Is the response acceptable and fit for the user?
2. "completeness": boolean - Is the response complete and addresses all aspects of the question?
3. "accuracy": boolean - Is the response accurate based on the provided context?
4. "proper_citation": boolean - Does the response properly cite chunk_ids when referencing information?
5. "feedback": string - Specific feedback to improve the response (if any flag is false, provide actionable feedback)

The response should be:
- Accurate and based only on the provided context
- Complete in addressing the user's question
- Properly cited with chunk_ids
- Clear and concise

Output ONLY valid JSON, no additional text or explanation.
"""

evaluator_agent = Agent(
    name="Response Evaluator",
    model=os.getenv("EVALUATOR_AGENT_MODEL", "gpt-4o-mini"),
    instructions=prompt,
    output_type=EvaluatorResponse
)

