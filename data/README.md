# Zillow Elasticsearch Data Ingestion

This directory contains scripts and configuration files for downloading Zillow property data and ingesting it into Elasticsearch.

## Files

- `variables.env.template` - Template for environment variables (copy to `variables.env`)
- `variables.env` - Environment variables for Elasticsearch connection and configuration (create from template)
- `requirements.txt` - Python dependencies
- `ingest-data.sh` - Shell script to set up virtual environment and run ingestion
- `ingest_data.py` - Main Python script for data ingestion
- `index_mappings.json` - Elasticsearch field mappings for the zillow-properties index
- `index_settings.json` - Elasticsearch index settings

## Setup

1. **Configure Environment Variables**:
   Copy the template and fill in your credentials:
   ```bash
   cp variables.env.template variables.env
   ```
   
   Then edit `variables.env` with your actual Elasticsearch credentials:
   ```bash
   ELASTICSEARCH_URL=https://your-elasticsearch-url:443
   ELASTICSEARCH_API_KEY=your-api-key
   INDEX_NAME=zillow-properties
   ```

2. **Run the Ingestion Script**:
   ```bash
   ./ingest-data.sh
   ```
   
   For a fresh download (delete existing CSV and re-download):
   ```bash
   ./ingest-data.sh -fresh
   ```

## How It Works

1. **Smart Download**: The script checks if the extracted data file exists locally. If not, it downloads the .7z archive from GitHub and extracts it.

2. **Index Management**: The script deletes any existing `zillow-properties` index and creates a new one with appropriate mappings and settings.

3. **Data Ingestion**: Uses Elasticsearch's parallel bulk helpers API for efficient data ingestion with configurable chunk size and thread count.

4. **Configuration Persistence**: After the first run, the script saves the index mappings and settings to JSON files for consistent reingestion.

## Configuration

All parameters are configurable via `variables.env`:

- `ELASTICSEARCH_URL` - Your Elasticsearch cluster URL
- `ELASTICSEARCH_API_KEY` - Your Elasticsearch API key
- `INDEX_NAME` - Name of the Elasticsearch index (default: zillow-properties)
- `CHUNK_SIZE` - Number of documents per bulk request (default: 500)
- `THREAD_COUNT` - Number of parallel threads for ingestion (default: 4)

Note: Shards and replicas are not configurable in Elasticsearch Serverless

## Reingestion

To reingest the data, simply run `./ingest-data.sh` again. The script will:
- Skip download if data already exists locally
- Delete and recreate the Elasticsearch index
- Reingest all data using the stored mappings and settings

## Data Source

The data is downloaded from: https://github.com/luminati-io/Zillow-dataset-samples/blob/main/zillow-properties-listing-information.7z
