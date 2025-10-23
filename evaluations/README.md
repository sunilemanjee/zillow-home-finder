# Query Parser Evaluation System

This directory contains the evaluation system for the MCP query parser, which assesses how well natural language queries are parsed into structured search parameters.

## Overview

The evaluation system uses **Azure AI Evaluation SDK** to provide comprehensive scoring of query parsing accuracy. It uses built-in evaluators (RelevanceEvaluator, GroundednessEvaluator, F1ScoreEvaluator) to assess parameter extraction quality and stores results in Elasticsearch for monitoring and analysis.

## Features

- **Azure AI Evaluation SDK**: Uses official Microsoft evaluation framework
- **Relevance Evaluation**: Measures if parsed parameters are relevant to the user query (1-5 scale)
- **Groundedness Evaluation**: Ensures parameters are grounded in the original query (1-5 scale)
- **F1 Score Evaluation**: Compares extracted parameters against ground truth (0-1 scale)
- **AI-Assisted Quality Metrics**: Uses GPT models for intelligent evaluation
- **Elasticsearch Storage**: All evaluation results stored with Azure SDK schema
- **OpenTelemetry Integration**: Full observability with trace correlation and custom metrics
- **Async Background Execution**: Non-blocking evaluation in search operations

## Configuration

### Environment Variables

Add to `variables.env`:

```bash
# Query Parser Evaluation Configuration
ENABLE_QUERY_EVAL=true

# Required for evaluation (reuse existing values)
ELASTICSEARCH_O11Y_URL=https://o11y-aa44fc.es.eastus.azure.elastic.cloud:443
ELASTICSEARCH_O11Y_API_KEY=your_api_key
EVAL_INDEX_NAME=evals
LLM_URL=https://your-llm-endpoint/v1
LLM_MODEL=gpt-4.1
LLM_API_KEY=your_llm_key
```

### MCP Tool Usage

When `ENABLE_QUERY_EVAL=true`, the MCP server exposes an `evaluate_query_parse` tool that:

1. **MUST be called after every `parse_query` call**
2. Takes the original query and parsed parameters as input
3. Returns detailed evaluation scores and issues
4. Automatically stores results in Elasticsearch

Example usage in Chainlit:
```
1. Call parse_query with: {"query": "homes with 2 beds under $500k"}
2. Call evaluate_query_parse with: {"query": "homes with 2 beds under $500k", "parsed_params": {...}}
```

## Evaluation Metrics

### Overall Score (0-1)
Weighted average of all evaluation components:
- Type Validation: 30%
- Range Validation: 20%
- Completeness: 30%
- Accuracy: 20%

### Individual Scores

1. **Type Validation Score**: Percentage of parameters with correct data types
2. **Range Validation Score**: Percentage of parameters within valid ranges
3. **Completeness Score**: How many extractable parameters were found
4. **Accuracy Score**: How well extracted parameters match the query intent

### Valid Parameters

Based on `data/search-template.mustache`:
- `query` (string): Semantic search text
- `distance` (number): Distance from location
- `distance_unit` (string): "mi" or "km"
- `latitude` (number): Latitude coordinate
- `longitude` (number): Longitude coordinate
- `bedrooms` (number): Minimum bedrooms
- `bathrooms` (number): Minimum bathrooms
- `tax` (number): Maximum tax value
- `maintenance` (number): Maximum HOA fee
- `square_footage` (number): Minimum square footage
- `home_price_min` (number): Minimum price
- `home_price_max` (number): Maximum price
- `feature` (string): Property feature

## Elasticsearch Schema

Evaluation results are stored with this structure:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "query": "original user query",
  "parsed_params": {...},
  "evaluation_scores": {
    "overall_score": 0.85,
    "type_validation_score": 0.9,
    "range_validation_score": 0.8,
    "completeness_score": 0.9,
    "accuracy_score": 0.8
  },
  "issues": ["bedrooms type mismatch", "distance out of range"],
  "trace_id": "abc123...",
  "span_id": "def456...",
  "evaluation_metadata": {
    "evaluator_version": "1.0",
    "evaluation_time_ms": 45.2
  }
}
```

## Testing

Run the test script to verify the integration:

```bash
cd evaluations
python test_evaluation.py
```

This will:
- Test evaluator initialization
- Verify Elasticsearch connection
- Run sample evaluations
- Display scoring results

## Monitoring

### Elasticsearch Queries

Find evaluation results:
```bash
# Get all evaluations
GET /evals/_search

# Get evaluations with low scores
GET /evals/_search
{
  "query": {
    "range": {
      "evaluation_scores.overall_score": {
        "lt": 0.7
      }
    }
  }
}

# Get evaluations by query pattern
GET /evals/_search
{
  "query": {
    "match": {
      "query": "bedrooms"
    }
  }
}
```

### OpenTelemetry Traces

All evaluations are traced with:
- `evaluator.overall_score`
- `evaluator.issues_count`
- `evaluator.success`
- `tool.execution_time_ms`

## Troubleshooting

### Tool Not Available
- Check `ENABLE_QUERY_EVAL=true` in variables.env
- Restart MCP server after changing environment variables
- Check logs for evaluator initialization errors

### Elasticsearch Connection Issues
- Verify `ELASTICSEARCH_O11Y_URL` and `ELASTICSEARCH_O11Y_API_KEY`
- Check network connectivity to Elasticsearch o11y cluster
- Ensure `EVAL_INDEX_NAME` index exists or can be created

### Low Evaluation Scores
- Review issues list in evaluation results
- Check parameter type mismatches
- Verify parameter ranges are reasonable
- Consider improving query parser prompts

## Implementation Details

### Files Modified
- `mcp/query_evaluator.py`: Core evaluation logic
- `mcp/server.py`: MCP tool integration
- `mcp/requirements.txt`: Added azure-ai-evaluation dependency
- `variables.env`: Added ENABLE_QUERY_EVAL flag

### Dependencies
- `azure-ai-evaluation==1.11.2`: Evaluation framework
- `elasticsearch>=8.0.0`: Storage backend
- `opentelemetry`: Observability integration

### Architecture
- Custom evaluator (not Azure built-ins) for structured parameter validation
- Conditional tool registration based on environment flag
- Comprehensive scoring with multiple validation layers
- Full observability integration with existing OpenTelemetry setup
