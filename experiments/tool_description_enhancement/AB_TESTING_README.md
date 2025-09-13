# A/B Testing Framework for Tool Description Enhancement

This framework provides comprehensive A/B testing capabilities to evaluate the impact of LLM-enhanced tool descriptions on search accuracy and relevance.

## Overview

The A/B testing framework compares search results between:
- **Baseline**: Original tool descriptions
- **Enhanced**: LLM-enhanced tool descriptions using Ollama Gemma3:12b

## Key Features

### 🎯 Comprehensive Metrics
- **Hit Rate at K**: Proportion of relevant tools in top-K results
- **Mean Reciprocal Rank (MRR)**: Ranking quality metric
- **Normalized Discounted Cumulative Gain (NDCG)**: Relevance-weighted ranking metric
- **Statistical Significance**: P-values and confidence intervals using paired t-tests

### 📊 Analysis Capabilities
- Category-based analysis (project management, file operations, etc.)
- Performance timing comparison
- Effect size calculation (Cohen's d)
- Detailed per-query breakdowns

### 🔄 Automated Testing
- Configurable test query sets
- Parallel search execution
- Automatic relevance marking
- Comprehensive reporting (JSON + Markdown)

## Quick Start

### 1. Setup Collections

```bash
cd /opt/stacks/lettatoolsselector/experiments/tool_description_enhancement

# Setup the A/B testing environment (creates ToolEnhanced collection)
python setup_ab_testing.py
```

This will:
- Verify baseline "Tool" collection exists
- Create "ToolEnhanced" collection with LLM-enhanced descriptions
- Populate enhanced collection with improved descriptions
- Verify both collections are ready

### 2. Run A/B Tests

```bash
# Run the complete A/B test suite
python ab_testing_framework.py
```

This will:
- Execute all test queries against both collections
- Calculate relevance metrics
- Perform statistical significance testing
- Generate comprehensive reports

### 3. Review Results

Results are saved to `ab_test_results/` directory:
- `ab_test_report_YYYYMMDD_HHMMSS.json` - Detailed data
- `ab_test_report_YYYYMMDD_HHMMSS.md` - Human-readable report

## Test Queries

The framework includes default test queries covering:

| Category | Example Query | Expected Tools |
|----------|---------------|----------------|
| Project Management | "create issue project management" | huly_create_issue, huly_create_subissue |
| File Operations | "file operations read write" | Read, Write, Edit |
| Memory Management | "memory management agent" | create_memory_block, update_memory_block |
| Search & Discovery | "search tools vector database" | find_tools, Grep, Glob |
| System Operations | "run commands terminal bash" | Bash, BashOutput |
| Web Operations | "web search fetch content" | WebSearch, WebFetch |
| Agent Management | "agent creation management" | create_agent, list_agents, modify_agent |

### Customizing Test Queries

Create `ab_test_queries.json` to customize test queries:

```json
[
  {
    "query": "your search query",
    "expected_tools": ["tool1", "tool2"],
    "category": "your_category",
    "priority": 1.0
  }
]
```

## Interpreting Results

### Key Metrics

- **Hit Rate at 5**: Percentage of queries where at least one relevant tool appears in top 5
- **MRR**: Average of 1/rank for first relevant tool (higher = better ranking)
- **NDCG at 5**: Relevance-weighted ranking quality (0-1, higher = better)

### Statistical Significance

- **P-Value < 0.05**: Statistically significant improvement
- **Effect Size**: Magnitude of improvement (Cohen's d)
  - 0.2 = small effect
  - 0.5 = medium effect
  - 0.8 = large effect

### Example Output

```
A/B TEST SUMMARY
==============================================================
Total Queries: 10
Categories: 7

KEY METRICS IMPROVEMENT:
  hit_rate_at_5  : +25.0% (p=0.0234) ✅
  mrr            : +18.5% (p=0.0456) ✅
  ndcg_at_5      : +22.1% (p=0.0123) ✅
```

## Framework Architecture

### Core Components

1. **ABTestFramework**: Main orchestration class
2. **SearchMetrics**: Relevance metric calculations
3. **TestQuery/TestResult**: Data structures for test cases and results
4. **Statistical Analysis**: Significance testing and effect size calculation

### Search Flow

```
Test Query → Enhanced Collection → Results → Mark Relevance → Calculate Metrics
           ↘ Baseline Collection  → Results → Mark Relevance → Calculate Metrics
                                                             ↓
                                                   Compare & Analyze
```

### Collections

- **Tool**: Baseline collection with original descriptions
- **ToolEnhanced**: Enhanced collection with LLM-improved descriptions

Both collections use identical schema but different description content for comparison.

## Configuration

### Environment Variables

```bash
# LLM Enhancement
ENABLE_LLM_ENHANCEMENT=true
OLLAMA_LLM_BASE_URL=http://100.81.139.20:11434/v1
OLLAMA_LLM_MODEL=gemma3:12b

# Weaviate Connection
WEAVIATE_URL=http://192.168.50.90:8080/
OPENAI_API_KEY=sk-...

# API Endpoints
API_BASE_URL=http://localhost:8020
```

### Framework Parameters

```python
framework = ABTestFramework(
    api_base_url="http://localhost:8020",
    weaviate_collection_enhanced="ToolEnhanced",
    weaviate_collection_baseline="Tool"
)
```

## Integration with Existing System

The framework integrates with:
- **Weaviate**: Vector database for tool search
- **LDTS API**: Tool search endpoints
- **Enhancement System**: LLM description improvement
- **Reranking System**: Optional result reranking

## Troubleshooting

### Common Issues

1. **"Collection does not exist"**
   - Run `setup_ab_testing.py` first
   - Check Weaviate connection

2. **"No tools loaded"**
   - Verify Letta API connection
   - Check `LETTA_API_URL` and `LETTA_PASSWORD`

3. **"LLM enhancement failed"**
   - Check Ollama server is running
   - Verify `OLLAMA_LLM_BASE_URL` is accessible

4. **"No significant results"**
   - May indicate enhancement isn't improving search
   - Try larger test query set
   - Check query relevance marking

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Setup Time**: ~5-15 minutes for collection creation and enhancement
- **Test Execution**: ~30 seconds for 10 test queries
- **Memory Usage**: Minimal - framework is stateless
- **LLM Calls**: One per tool during setup (cached afterward)

## Future Enhancements

- [ ] Automated relevance judgment using LLM
- [ ] Multi-model comparison (different enhancement models)
- [ ] Real-time A/B testing integration
- [ ] User behavior tracking integration
- [ ] Bayesian statistical analysis

## Contributing

When adding new test queries:
1. Ensure expected tools are actually relevant
2. Cover diverse tool categories
3. Include edge cases and ambiguous queries
4. Test with domain experts when possible