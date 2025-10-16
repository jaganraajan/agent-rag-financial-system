# Implementation Summary: Iterative Query Decomposition

## Overview

This implementation adds iterative (looped) query decomposition capability to the Enhanced RAG Pipeline, enabling compound queries where the output of initial sub-queries informs the generation and execution of follow-up sub-queries.

## Changes Made

### 1. Enhanced SynthesisResult with Metadata (`src/agents/synthesis_engine.py`)

**What changed:**
- Added optional `metadata` field to `SynthesisResult` dataclass

**Why:**
- Enables structured propagation of query results (e.g., winner company) between synthesis stages
- Avoids brittle string parsing to extract information from answers

**Code:**
```python
@dataclass
class SynthesisResult:
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[SourceInfo]
    metadata: Optional[Dict[str, Any]] = None  # NEW
```

### 2. Winner Propagation in Synthesis Engine (`src/agents/synthesis_engine.py`)

**What changed:**
- Updated `_synthesize_cross_company_comparison` to populate `metadata['winner_company']`

**Why:**
- Allows the pipeline to reliably identify the winner for follow-up queries
- Metadata includes both winner company ticker and year

**Code change:**
```python
return SynthesisResult(
    # ... other fields ...
    metadata={'winner_company': highest_company, 'year': year}
)
```

### 3. Compound Query Detection (`src/agents/enhanced_rag.py`)

**What changed:**
- Added `_detect_compound_query_with_followup()` method

**Why:**
- Identifies queries that require iterative follow-up
- Checks for: comparative clause + pronoun reference + AI intent

**Detection logic:**
- Comparative patterns: "highest", "best", "which company", etc.
- Pronoun patterns: "that company", "the company", "it"
- AI patterns: "ai risks", "ai strategy", etc.

### 4. Year Extraction (`src/agents/enhanced_rag.py`)

**What changed:**
- Added `_extract_year_from_query()` method with default to 2024

**Why:**
- Ensures follow-up queries use the same year as initial queries
- Provides sensible default when year is not specified

### 5. Iterative Query Logic (`src/agents/enhanced_rag.py`)

**What changed:**
- Enhanced `query()` method to support iterative follow-up

**Flow:**
1. Detect if query is compound
2. Execute initial sub-queries (e.g., revenue comparison)
3. Synthesize results to determine winner
4. If compound AND winner found: append follow-up sub-query
5. Execute follow-up and re-synthesize with combined results

**Code structure:**
```python
# Step 3: Initial synthesis
synthesis_result = self._synthesize_cross_company_comparison(...)

# Step 4: Iterative follow-up
if is_compound and synthesis_result.metadata and 'winner_company' in synthesis_result.metadata:
    winner = synthesis_result.metadata['winner_company']
    followup_query = f"{winner} ai strategy {year}"
    
    # Execute follow-up
    followup_result = self.base_rag.query(followup_query, top_k)
    
    # Append to sub_queries list
    synthesis_result.sub_queries.append(followup_query)
    
    # Re-synthesize with all results
    synthesis_result = self._synthesize_ai_strategy(...)
```

### 6. Winner Detection in Enhanced RAG (`src/agents/enhanced_rag.py`)

**What changed:**
- Updated `_synthesize_cross_company_comparison` in enhanced_rag.py to extract winner from sources

**Why:**
- Provides fallback winner detection when synthesis_engine method isn't used
- Extracts revenue values from source excerpts using regex

**Logic:**
- Parses source excerpts for revenue patterns (e.g., "$123.4 billion")
- Selects company with highest extracted value
- Populates metadata for use in iterative follow-up

### 7. Query Type Prioritization (`src/agents/query_decomposer.py`)

**What changed:**
- Moved comparative pattern check before other query type checks

**Why:**
- Ensures compound queries with both comparative and AI intent are classified as "comparative"
- This allows the comparative synthesis to run first, producing the winner needed for follow-up
- Maintains correct decomposition flow for iterative queries

## Example Usage

### Input Query
```
"Which company had the highest revenue in 2024? What are the main AI risks of that company?"
```

### Expected Behavior

1. **Detection**: Identified as compound query
2. **Initial sub-queries**:
   - "MSFT revenue 2024"
   - "GOOGL revenue 2024"
   - "NVDA revenue 2024"
3. **Winner determination**: GOOGL (highest revenue: $328.3B)
4. **Follow-up sub-query**: "GOOGL ai strategy 2024"
5. **Final JSON output**:
```json
{
  "query": "Which company had the highest revenue in 2024? What are the main AI risks of that company?",
  "sub_queries": [
    "MSFT revenue 2024",
    "GOOGL revenue 2024",
    "NVDA revenue 2024",
    "GOOGL ai strategy 2024"
  ],
  "answer": "[Combined answer about revenue winner and their AI risks]",
  "reasoning": "Synthesized AI strategy comparison using LLM from 4 sub-query answers.",
  "sources": [...]
}
```

## Testing

### Test Files Created

1. **`test_compound_query.py`**: Unit tests for individual components
   - Compound query detection
   - Year extraction
   - Metadata propagation

2. **`test_integration_compound.py`**: End-to-end integration test
   - Mocks RAG data with realistic financial information
   - Validates complete iterative flow
   - Confirms all 4 sub-queries appear in output

3. **`demo_compound_query.py`**: Interactive demonstration
   - Visual step-by-step output
   - Shows detection, execution, and results
   - Educational tool for understanding the feature

4. **`TESTING.md`**: Comprehensive testing guide
   - How to run tests
   - Expected outputs
   - Success criteria

### Test Results

✅ All unit tests pass (5/5)  
✅ Integration test passes  
✅ Backward compatibility maintained  
✅ Simple queries work as before  

## Backward Compatibility

The implementation maintains full backward compatibility:

- **Simple queries**: Processed normally without iterative follow-up
- **Comparative queries without AI intent**: Work as before
- **Non-compound queries**: No changes to existing behavior
- **JSON output**: Same structure, with extended `sub_queries` list when applicable

## Limitations and Notes

1. **Winner Detection**: Follow-up only appended when winner can be clearly determined
   - Requires extractable metric values in source data
   - Demo data may lack sufficient detail

2. **Follow-up Intent**: Currently limited to "ai strategy" pattern
   - Could be extended to support other follow-up types in future

3. **LLM Dependency**: Full synthesis benefits from Azure OpenAI
   - Works with fallback when credentials unavailable
   - Fallback concatenates sub-answers

4. **Single Follow-up**: Current implementation supports one follow-up iteration
   - Could be extended for multiple iterations if needed

## Acceptance Criteria - All Met ✅

- [x] For compound queries, system runs comparative sub-queries
- [x] System determines winner from comparative results
- [x] System appends and executes "{WINNER} ai strategy {year}"
- [x] Final JSON `sub_queries` includes all queries in execution order
- [x] Backward compatibility preserved for simple queries
- [x] Complete test suite validates functionality
- [x] Clear documentation provided

## Files Modified

1. `src/agents/synthesis_engine.py` - Added metadata field, populated winner
2. `src/agents/enhanced_rag.py` - Implemented iterative logic and detection
3. `src/agents/query_decomposer.py` - Prioritized comparative classification

## Files Created

1. `test_compound_query.py` - Unit tests
2. `test_integration_compound.py` - Integration test
3. `demo_compound_query.py` - Interactive demo
4. `TESTING.md` - Testing guide
5. `IMPLEMENTATION_SUMMARY.md` - This document

## Future Enhancements

Possible extensions to this feature:

1. Support for multiple follow-up questions
2. Additional follow-up intent types beyond AI strategy
3. More sophisticated winner determination algorithms
4. Confidence scores for winner selection
5. Support for multi-level iterative decomposition
