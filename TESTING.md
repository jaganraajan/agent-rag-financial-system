# Testing Guide for Compound Query Decomposition

## Overview

This document describes how to test the new iterative (looped) query decomposition feature.

## Test Suite

### 1. Unit Tests (`test_compound_query.py`)

Tests individual components:
- Compound query detection
- Year extraction
- Synthesis metadata (winner propagation)

Run with:
```bash
python test_compound_query.py
```

Expected output: All 5 tests should pass.

### 2. Integration Tests (`test_integration_compound.py`)

Tests the complete end-to-end flow with mocked RAG data.

Run with:
```bash
python test_integration_compound.py
```

Expected output:
- Query is correctly identified as compound
- 3 comparative revenue sub-queries are executed
- Winner (GOOGL) is determined from mock data
- Follow-up AI strategy query is appended
- Final JSON includes all 4 sub-queries

**Sample output:**
```json
{
  "query": "Which company had the highest revenue in 2024? What are the main AI risks of that company?",
  "sub_queries": [
    "GOOGL revenue 2024",
    "MSFT revenue 2024",
    "NVDA revenue 2024",
    "GOOGL ai strategy 2024"
  ],
  "sources": [...]
}
```

## Key Features Verified

1. **Compound Query Detection**: System correctly identifies queries with both:
   - Comparative clause (e.g., "highest revenue")
   - Follow-up clause with pronoun reference (e.g., "that company")
   - AI-related intent (e.g., "AI risks")

2. **Iterative Execution**:
   - Initial comparative sub-queries are executed
   - Winner is determined from results
   - Follow-up sub-query is dynamically generated and executed
   - All sub-queries appear in final output

3. **Backward Compatibility**:
   - Simple queries without follow-up work as before
   - Comparative queries without AI intent work as before
   - Only compound queries trigger iterative behavior

## Test Queries

### Compound Queries (Triggers Iterative Flow)
- "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
- "What company had the best performance in 2024? What are the AI strategies of that company?"

### Simple Queries (No Iterative Flow)
- "Which company had the highest operating margin in 2023?"
- "Compare Microsoft and Google revenue"
- "What are NVIDIA's main business activities?"

## With Real Data

To test with actual SEC filings:

1. Ensure demo filings or real SEC data is available
2. Process the filings:
   ```bash
   python main.py rag --process --input-dir demo_filings
   ```

3. Test compound query:
   ```bash
   python -c "
   from src.agents.enhanced_rag import EnhancedRAGPipeline
   import json
   
   pipeline = EnhancedRAGPipeline()
   query = 'Which company had the highest revenue in 2024? What are the main AI risks of that company?'
   result = pipeline.query(query, return_json=True)
   print(json.dumps(result, indent=2))
   "
   ```

**Note**: Demo filings are minimal and may not contain actual financial data. For best results, use real SEC 10-K filings with financial information.

## Success Criteria

✅ Integration test passes  
✅ Compound queries generate 4+ sub-queries (initial + follow-up)  
✅ Simple queries maintain backward compatibility  
✅ Sub-queries list in JSON output includes all executed queries  
✅ Winner company is correctly identified from comparative results  
✅ Follow-up query uses winner ticker and correct year  

## Limitations

- Follow-up is only appended when a clear winner can be determined
- Demo data may not contain sufficient information to extract winners
- Without Azure OpenAI credentials, answers use fallback concatenation
- Follow-up queries are currently limited to "ai strategy" intent
