# Agent RAG Financial System – Design Overview

## 1. Chunking Strategy
The system uses section-aware, semantic chunking tailored for SEC 10-K filings. Documents are parsed with BeautifulSoup, and split into chunks of 50–1000 tokens, preserving natural boundaries (paragraphs, sentences, and 10-K sections like PART I, Item 1A, etc.). Financial tables are detected and chunked separately. Each chunk is enriched with metadata: company, year, section title/type, and token count. This approach ensures high retrieval precision and context relevance for downstream analysis.

## 2. Embedding Model Choice
**Azure OpenAI text-embedding-ada-002** is used for chunk embeddings. This model is chosen for its:
- High dimensionality (1536-dim) and semantic accuracy
- Robustness for financial and technical text
- Cost-effectiveness and speed for large-scale document processing
If offline, a deterministic mock embedding is used for testing. ChromaDB stores all embeddings for fast similarity search and filtering.

## 3. Agent & Query Decomposition Approach
LangGraph powers the agent’s query decomposition. Complex user questions are automatically broken into targeted sub-queries (e.g., per company, per year, per metric). The agent detects query type (simple, comparative, YoY, segment, AI strategy) and generates sub-queries accordingly. Each sub-query retrieves relevant chunks from the vector store, ensuring multi-step, multi-entity analysis.

## 4. LLM-Powered Synthesis
After retrieval, the top chunks are passed to an LLM (Azure OpenAI GPT-4o-mini) with a financial analyst prompt. The LLM synthesizes a concise, context-aware answer, citing specific values and reasoning. For comparative and multi-company queries, the agent aggregates sub-query answers and prompts the LLM to summarize differences and trends. This approach maximizes answer accuracy and interpretability, even for complex financial questions.

---
**Summary:**
The system combines robust chunking, state-of-the-art embeddings, intelligent query decomposition, and LLM synthesis to deliver precise, explainable answers from SEC filings for Google, Microsoft, and NVIDIA.
