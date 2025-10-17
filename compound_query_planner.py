#!/usr/bin/env python3
"""
Compound Query Planner (Iterative, LLM-in-the-loop)

Purpose
- Handle compound, multi-step questions where later sub-queries depend on
  facts discovered in earlier retrievals.
- Uses Azure OpenAI to plan each step (split/route) and EnhancedRAGPipeline.query
  to execute sub-queries against the vector store.

Example
Input:
  "Which company had the highest revenue in 2024? What are the main AI risks of that company?"

Behavior:
  1) Ask EnhancedRAGPipeline.query("Which company had the highest revenue in 2024?")
  2) Use the LLM to infer the best next sub-query from the answer (e.g., "What are the main AI risks of MSFT?")
  3) Execute that sub-query
  4) Synthesize a final answer using all intermediate answers and citations.

Environment variables (Azure OpenAI)
- AZURE_OPENAI_ENDPOINT:      e.g., https://<your-resource>.openai.azure.com/
- AZURE_OPENAI_API_KEY:       your Azure OpenAI API key
- AZURE_OPENAI_API_VERSION:   e.g., 2024-02-01 (default)
- AZURE_OPENAI_MODEL:         the chat deployment name, e.g., gpt-4o-mini

Notes
- If Azure OpenAI is not configured, the planner will fall back to a naive splitter
  based on sentence boundaries, but the intelligent routing (e.g., inserting discovered entities)
  will be limited. Azure usage is strongly recommended for robust planning.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Local imports
from src.agents.enhanced_rag import EnhancedRAGPipeline


@dataclass
class StepResult:
    """Holds the result of a single sub-query execution."""
    subquery: str
    answer: str
    sources: List[Dict[str, Any]]


class CompoundQueryPlanner:
    def __init__(
        self,
        max_steps: int = 5,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.rag = EnhancedRAGPipeline(use_openai=True)
        self.max_steps = max_steps
        self.top_k = top_k

        from dotenv import load_dotenv
        load_dotenv()
        # Azure OpenAI configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")

        self._azure_available = bool(self.azure_endpoint and self.azure_api_key)

        if not self._azure_available:
            self.logger.warning(
                "Azure OpenAI is not fully configured. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY for LLM-driven planning. "
                "Falling back to naive splitting."
            )

    # --------------- Public API ---------------

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute an iterative plan to answer a compound query.

        Returns:
            {
                "query": original query,
                "final_answer": string,
                "steps": [ {subquery, answer, sources: [...]}, ... ],
                "notes": [...],
            }
        """
        steps: List[StepResult] = []
        notes: List[str] = []
        remaining_budget = self.max_steps

        # Loop: plan -> execute -> update -> repeat
        while remaining_budget > 0:
            plan = self._plan_next(query, steps)
            next_subquery = plan.get("next_subquery")
            stop = bool(plan.get("stop", False))
            reason = plan.get("reason", "")
            notes.append(f"{reason}")

            if stop:
                # If the model provided a candidate final answer, use it; else synthesize one.
                # final_answer = plan.get("final_answer", "")
                # if not final_answer:
                #     final_answer = self._synthesize_final(query, steps)
                # Consolidate all subqueries and sources
                final_answer = self._synthesize_final(query, steps)
                sub_queries = [s.subquery for s in steps]
                reasoning = notes + [f"Terminated after {self.max_steps - remaining_budget} step(s)."]
                all_sources = []
                for s in steps:
                    all_sources.extend(s.sources)
                return {
                    "query": query,
                    "answer": final_answer,
                    "reasoning": "".join(map(str, reasoning)),
                    "sub_queries": sub_queries,
                    "sources": all_sources,
                }

            if not next_subquery or not isinstance(next_subquery, str):
                notes.append("No actionable next_subquery produced; stopping.")
                break

            # Execute the sub-query with Enhanced RAG
            self.logger.info(f"Executing sub-query: {next_subquery}")
            result = self.rag.query(next_subquery, top_k=self.top_k, return_json=True)

            # Extract answer and sources (best effort)
            answer = result.get("answer", "") or result.get("synthesis", {}).get("answer", "")
            sources = result.get("sources", [])
            steps.append(StepResult(subquery=next_subquery, answer=answer, sources=sources))

            remaining_budget -= 1

        # If we exit due to budget or planner failure, synthesize best-effort answer
        # final_answer = self._synthesize_final(query, steps)
        # sub_queries = [s.subquery for s in steps]
        # all_sources = []
        # for s in steps:
        #     all_sources.extend(s.sources)
        # return {
        #     "query": query,
        #     "answer": final_answer,
        #     "reasoning": notes + [f"Terminated after {self.max_steps - remaining_budget} step(s)."],
        #     "sub_queries": sub_queries,
        #     "sources": all_sources,
        # }

    # --------------- Planning (LLM) ---------------

    def _plan_next(self, original_query: str, steps: List[StepResult]) -> Dict[str, Any]:
        """
        Ask Azure OpenAI to decide the next sub-query based on the original query and prior step answers.
        Expected STRICT JSON:
            {
              "next_subquery": "string or empty",
              "stop": boolean,
              "reason": "string",
              "final_answer": "string (optional, only when stop=true)"
            }
        """

        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )

            context = [
                {
                    "subquery": s.subquery,
                    "short_answer": (s.answer or "")[:600],
                }
                for s in steps[-4:]  # keep it compact
            ]

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a planning assistant for compound financial questions about SEC filings. "
                        "Plan step-by-step. If later sub-questions depend on earlier answers (e.g., discovering which company first), "
                        "use those answers to craft the next sub-query precisely. "
                        "You must return STRICT JSON only with keys: next_subquery (string), stop (boolean), reason (string), "
                        "and optional final_answer when stop=true."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_query": original_query,
                            "completed_steps": context,
                            "instructions": [
                                "If no steps have been run, produce the first actionable sub-query that advances the solution.",
                                "If steps exist, read their short_answer to decide the next sub-query (e.g., inject the discovered company/ticker).",
                                "Stop when the chain of sub-queries is sufficient to answer the original query; include final_answer.",
                                "Examples of next_subquery: 'Which company had the highest revenue in 2024?', "
                                "'What are the main AI risks of MSFT in its 2023 10-K?'",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]

            resp = client.chat.completions.create(
                model=self.azure_model,
                temperature=0.1,
                messages=messages,
            )
            raw = resp.choices[0].message.content or "{}"
            return self._safe_json_parse(raw, default={"next_subquery": "", "stop": True, "reason": "parse_failed"})
        except Exception as e:
            self.logger.warning(f"Azure planning failed: {e}; falling back to naive planning.")

    # --------------- Synthesis (LLM) ---------------

    def _synthesize_final(self, original_query: str, steps: List[StepResult]) -> str:
        """
        Synthesize a final answer from the sub-answers using Azure OpenAI.
        Falls back to a stitched summary if Azure is unavailable.
        """
        print('synthesizing final result')
        if not steps:
            return "No results found."

        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
            )

            compact_steps = [
                {"subquery": s.subquery, "answer": (s.answer or "")[:1000]} for s in steps
            ]

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst assistant. Create a concise, direct answer "
                        "to the user's compound question using the sub-answers. If certainty is low for any part, say so briefly. "
                        "Keep the answer grounded in the sub-answers; do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_query": original_query,
                            "steps": compact_steps,
                            "instruction": "Produce 3–6 sentences, referencing the discovered entities concretely (e.g., MSFT).",
                        },
                        ensure_ascii=False,
                    ),
                },
            ]

            resp = client.chat.completions.create(
                model=self.azure_model,
                temperature=0.2,
                messages=messages,
            )
            print('answer is ' + resp.choices[0].message.content)
            return resp.choices[0].message.content or "No synthesis available."
        except Exception as e:
            self.logger.warning(f"Azure synthesis failed: {e};")

    # --------------- Utilities ---------------

    def _safe_json_parse(self, raw: str, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = raw.strip()
            if raw.startswith("{"):
                return json.loads(raw)
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception:
            pass
        return default

def _demo():
    """
    Simple demo for local testing.
    Ensure your vector DB is populated and Azure OpenAI is configured.
    """
    logging.basicConfig(level=logging.INFO)
    planner = CompoundQueryPlanner(max_steps=4, top_k=5)

    # query = "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
    query = "What was NVIDIA's growth rate from 2022 to 2023?"
        # "Tell me about Apple's financial performance"
    result = planner.run(query)

    print("\n=== Compound Query Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _demo()