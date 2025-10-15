# MCP Home Finder Server

A Model Context Protocol (MCP) server that helps users find homes stored in Elasticsearch by processing natural language queries, geocoding locations with Azure Maps, and executing searches using Elasticsearch search templates.

## Features

- **Natural Language Processing**: Uses LLM to parse complex queries into structured search parameters
- **Geocoding**: Azure Maps integration for converting location names to coordinates
- **Elasticsearch Integration**: Executes searches using pre-configured search templates
- **SSE Transport**: Server-Sent Events over HTTP for real-time streaming responses
- **Comprehensive Search**: Supports bedrooms, bathrooms, price ranges, square footage, features, and location-based searches

## Setup

### Prerequisites

- Python 3.8 or higher
- Elasticsearch instance with Zillow property data
- Azure Maps subscription key
- LLM API access (configured in variables.env)

### Installation

1. **Install dependencies**:
   ```bash
   cd mcp
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   - Copy `../variables.env.template` to `../variables.env`
   - Fill in your actual credentials:
     ```env
     # Elasticsearch Configuration
     ELASTICSEARCH_URL=https://your-elasticsearch-url:443
     ELASTICSEARCH_API_KEY=your-elasticsearch-api-key
     INDEX_NAME=zillow-properties
     
     # LLM Configuration
     LLM_URL=https://your-llm-endpoint/v1/chat/completions
     LLM_MODEL=gpt-4
     LLM_API_KEY=your-llm-api-key
     
     # Azure Maps Configuration
     AZURE_MAPS_SUBSCRIPTION_KEY=your-azure-maps-key
     
     # Search Template Configuration
     SEARCH_TEMPLATE_NAME=zillow-property-search
     ```

### Running the Server

```bash
# From the mcp directory
python -m mcp
```

The server will start and listen for MCP connections over SSE transport.

## Usage

### Tool: search_homes

The server provides a single tool called `search_homes` that accepts natural language queries.

#### Examples

**Location-based searches**:
- "homes within 10 miles of Orlando FL with 2 beds"
- "3 bedroom house in Miami under $500k"
- "condos near downtown Seattle"

**Feature-based searches**:
- "house with pool and garage"
- "apartment with gym and parking"
- "home with updated kitchen"

**Price and size filters**:
- "homes under $300k with at least 1500 sqft"
- "2 bedroom house between $200k and $400k"
- "large family home over 3000 square feet"

**Combined searches**:
- "3 bedroom house with pool within 15 miles of Tampa FL under $600k"
- "modern condo in downtown with gym and parking"

### Search Parameters

The system automatically extracts these parameters from your queries:

- **query**: Semantic search text for property descriptions
- **distance**: Distance in miles from a location
- **latitude/longitude**: Coordinates for location-based searches
- **bedrooms**: Minimum number of bedrooms
- **bathrooms**: Minimum number of bathrooms
- **tax**: Maximum tax assessed value
- **maintenance**: Maximum HOA/maintenance fee
- **square_footage**: Minimum square footage
- **home_price_min/max**: Price range
- **feature**: Specific property features (pool, garage, etc.)

## Architecture

```
User Query → Query Parser (LLM) → Geocoder (Azure Maps) → Search Service (Elasticsearch) → Results
```

1. **Query Parser**: Uses LLM to extract structured parameters from natural language
2. **Geocoder**: Converts location names to latitude/longitude coordinates
3. **Search Service**: Executes Elasticsearch search template with extracted parameters
4. **Results**: Returns formatted home listings with relevant details

## Error Handling

The server includes comprehensive error handling:

- **LLM failures**: Falls back to regex-based parsing
- **Geocoding failures**: Continues search without location constraints
- **Elasticsearch errors**: Returns user-friendly error messages
- **Network timeouts**: Graceful degradation with informative responses

## Development

### Project Structure

```
mcp/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point
├── server.py            # Main MCP server implementation
├── query_parser.py      # LLM-based query parsing
├── geocoder.py          # Azure Maps geocoding
├── search_service.py    # Elasticsearch integration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features

1. **New search parameters**: Update the system prompt in `query_parser.py`
2. **Additional geocoding providers**: Extend `geocoder.py` with new providers
3. **Enhanced result formatting**: Modify `search_service.py`
4. **New tools**: Add to `server.py` tool definitions

## Troubleshooting

### Common Issues

1. **"No homes found"**: Check if your Elasticsearch index contains data and the search template is properly configured
2. **Geocoding failures**: Verify your Azure Maps subscription key and quota
3. **LLM parsing errors**: Check your LLM API credentials and endpoint
4. **Connection errors**: Ensure Elasticsearch is accessible and credentials are correct

### Logs

The server logs important events and errors. Check the console output for debugging information.

## License

This project is part of the Zillow Home Finder system.
