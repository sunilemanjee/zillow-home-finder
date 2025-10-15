#!/usr/bin/env python3
"""
Zillow Elasticsearch Data Ingestion Script

This script downloads Zillow property data from a .7z archive, creates an Elasticsearch index
with appropriate mappings and settings, and ingests the data using parallel bulk helpers API.
"""

import os
import json
import logging
import requests
import py7zr
import csv
import pandas as pd
from typing import Dict, Any, List, Optional
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('../variables.env')

# Configuration from environment variables
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')
ELASTICSEARCH_API_KEY = os.getenv('ELASTICSEARCH_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME', 'zillow-properties')
SEARCH_TEMPLATE_NAME = os.getenv('SEARCH_TEMPLATE_NAME', 'zillow-property-search')
DATA_URL = os.getenv('DATA_URL')
ARCHIVE_NAME = os.getenv('ARCHIVE_NAME', 'zillow-properties-listing-information.7z')
EXTRACTED_FILE_NAME = os.getenv('EXTRACTED_FILE_NAME', 'zillow-properties-listing-information.json')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
THREAD_COUNT = int(os.getenv('THREAD_COUNT', 4))

# File paths for index configuration
MAPPINGS_FILE = 'index_mappings.json'
SETTINGS_FILE = 'index_settings.json'
SEARCH_TEMPLATE_FILE = 'search-template.mustache'


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = ['ELASTICSEARCH_URL', 'ELASTICSEARCH_API_KEY', 'DATA_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("Environment validation passed")


def download_and_extract_data() -> None:
    """Download and extract data if not already present locally."""
    # Check if fresh download is requested via environment variable
    fresh_download = os.getenv('FRESH_DOWNLOAD', 'false').lower() == 'true'
    
    if os.path.exists(EXTRACTED_FILE_NAME) and not fresh_download:
        logger.info(f"Extracted data file '{EXTRACTED_FILE_NAME}' already exists, skipping download")
        return
    
    if fresh_download and os.path.exists(EXTRACTED_FILE_NAME):
        logger.info(f"Fresh download requested - removing existing file: {EXTRACTED_FILE_NAME}")
        os.remove(EXTRACTED_FILE_NAME)
    
    logger.info(f"Downloading data from {DATA_URL}")
    
    # Download the archive
    try:
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()
        
        with open(ARCHIVE_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded archive: {ARCHIVE_NAME}")
        
    except requests.RequestException as e:
        logger.error(f"Failed to download data: {e}")
        raise
    
    # Extract the archive
    logger.info(f"Extracting archive: {ARCHIVE_NAME}")
    try:
        with py7zr.SevenZipFile(ARCHIVE_NAME, mode='r') as z:
            z.extractall()
        
        logger.info("Archive extracted successfully")
        
    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        raise
    
    # Remove the archive after extraction
    try:
        os.remove(ARCHIVE_NAME)
        logger.info(f"Removed archive file: {ARCHIVE_NAME}")
    except OSError as e:
        logger.warning(f"Failed to remove archive file: {e}")


def analyze_data_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the data structure to create appropriate Elasticsearch mappings."""
    if not data:
        return {}
    
    # Sample a few records to understand the structure
    sample_size = min(10, len(data))
    sample_data = data[:sample_size]
    
    mappings = {"properties": {}}
    
    for record in sample_data:
        for field, value in record.items():
            if field not in mappings["properties"]:
                # Determine field type based on value
                if isinstance(value, str):
                    # Check if it looks like a date
                    if any(keyword in field.lower() for keyword in ['date', 'time', 'created', 'updated']):
                        mappings["properties"][field] = {"type": "date"}
                    else:
                        mappings["properties"][field] = {"type": "text"}
                elif isinstance(value, int):
                    mappings["properties"][field] = {"type": "integer"}
                elif isinstance(value, float):
                    mappings["properties"][field] = {"type": "float"}
                elif isinstance(value, bool):
                    mappings["properties"][field] = {"type": "boolean"}
                else:
                    mappings["properties"][field] = {"type": "text"}
    
    return mappings


def load_or_create_mappings() -> Dict[str, Any]:
    """Load existing mappings or create new ones based on data analysis."""
    if os.path.exists(MAPPINGS_FILE):
        logger.info(f"Loading existing mappings from {MAPPINGS_FILE}")
        with open(MAPPINGS_FILE, 'r') as f:
            return json.load(f)
    
    logger.info("No existing mappings found, analyzing data structure...")
    
    # Load data to analyze structure
    data = []
    with open(EXTRACTED_FILE_NAME, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        row_count = 0
        for row in csv_reader:
            row_count += 1
            data.append(row)
            # Only analyze first 10 rows for mapping creation
            if row_count >= 10:
                break
    
    mappings = analyze_data_structure(data)
    
    # Save mappings for future use
    with open(MAPPINGS_FILE, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    logger.info(f"Created and saved mappings to {MAPPINGS_FILE}")
    return mappings


def load_or_create_settings() -> Dict[str, Any]:
    """Load existing settings or create default ones."""
    if os.path.exists(SETTINGS_FILE):
        logger.info(f"Loading existing settings from {SETTINGS_FILE}")
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    
    logger.info("Creating default index settings...")
    
    settings = {
        "index": {
            "analysis": {
                "analyzer": {
                    "address_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop"]
                    }
                }
            }
        }
    }
    
    # Save settings for future use
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    
    logger.info(f"Created and saved settings to {SETTINGS_FILE}")
    return settings


def create_elasticsearch_client() -> Elasticsearch:
    """Create and return an Elasticsearch client."""
    try:
        es = Elasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
            verify_certs=True,
            request_timeout=60
        )
        
        # Test connection
        info = es.info()
        logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
        return es
        
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        raise


def create_index(es: Elasticsearch) -> None:
    """Delete existing index if it exists and create a new one with mappings and settings."""
    # Delete existing index if it exists
    if es.indices.exists(index=INDEX_NAME):
        logger.info(f"Deleting existing index: {INDEX_NAME}")
        es.indices.delete(index=INDEX_NAME)
    
    # Load mappings and settings
    mappings = load_or_create_mappings()
    settings = load_or_create_settings()
    
    # Create the index
    logger.info(f"Creating new index: {INDEX_NAME}")
    es.indices.create(
        index=INDEX_NAME,
        mappings=mappings,
        settings=settings
    )
    
    logger.info(f"Index '{INDEX_NAME}' created successfully")


def create_search_template(es: Elasticsearch) -> None:
    """Create or update the search template in Elasticsearch."""
    if not os.path.exists(SEARCH_TEMPLATE_FILE):
        logger.warning(f"Search template file '{SEARCH_TEMPLATE_FILE}' not found, skipping template creation")
        return
    
    logger.info(f"Creating search template from {SEARCH_TEMPLATE_FILE}")
    
    try:
        # Read the search template
        with open(SEARCH_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Note: We don't validate as JSON since this is a Mustache template
        # with conditional logic that isn't valid JSON
        logger.info("Template content loaded successfully (Mustache template validation skipped)")
        
        # Create the search template
        template_name = SEARCH_TEMPLATE_NAME
        
        # Delete existing template if it exists
        try:
            es.delete_script(id=template_name)
            logger.info(f"Deleted existing search template: {template_name}")
        except:
            # Template doesn't exist, which is fine
            pass
        
        # Create the new search template
        es.put_script(
            id=template_name,
            body={
                "script": {
                    "lang": "mustache",
                    "source": template_content
                }
            }
        )
        
        logger.info(f"Search template '{template_name}' created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create search template: {e}")
        raise


def test_search_template(es: Elasticsearch) -> None:
    """Test the search template with a simple query."""
    template_name = SEARCH_TEMPLATE_NAME
    
    try:
        # Test the search template with a simple query
        test_params = {
            "query": "modern house",
            "bedrooms": 2,
            "bathrooms": 1,
            "home_price_max": 500000
        }
        
        logger.info("Testing search template with sample parameters...")
        
        # Execute the search template
        response = es.search_template(
            index=INDEX_NAME,
            body={
                "id": template_name,
                "params": test_params
            }
        )
        
        hits = response.get('hits', {}).get('total', {})
        if isinstance(hits, dict):
            total_hits = hits.get('value', 0)
        else:
            total_hits = hits
        
        logger.info(f"Search template test successful - found {total_hits} matching documents")
        
    except Exception as e:
        logger.warning(f"Search template test failed: {e}")
        # Don't raise here as this is just a test


def parse_csv_line(line: str, headers: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a CSV line that may contain complex JSON data."""
    try:
        # Use Python's csv module for more robust parsing
        import io
        csv_reader = csv.reader(io.StringIO(line), quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        
        try:
            fields = next(csv_reader)
        except StopIteration:
            return None
        
        # Create dictionary with headers
        if len(fields) >= len(headers):
            return dict(zip(headers, fields[:len(headers)]))
        else:
            # Pad with None values if we have fewer fields than headers
            padded_fields = fields + [None] * (len(headers) - len(fields))
            return dict(zip(headers, padded_fields))
            
    except Exception as e:
        logger.warning(f"Error parsing CSV line: {e}")
        return None


def ingest_data(es: Elasticsearch) -> None:
    """Ingest data into Elasticsearch using parallel bulk helpers API."""
    logger.info(f"Starting data ingestion into index: {INDEX_NAME}")
    
    # Load data from CSV using custom parser
    logger.info("Loading CSV data with custom parser...")
    data = []
    
    try:
        with open(EXTRACTED_FILE_NAME, 'r', encoding='utf-8') as f:
            # Read the header first
            header_line = f.readline().strip()
            headers = [h.strip() for h in header_line.split(',')]
            logger.info(f"Found {len(headers)} columns in CSV")
            
            row_count = 0
            for line in f:
                row_count += 1
                try:
                    # Parse the CSV line
                    row_data = parse_csv_line(line.strip(), headers)
                    
                    if row_data:
                        # Convert numeric fields
                        numeric_fields = ['price', 'bedrooms', 'bathrooms', 'livingArea', 'lotSize', 'zestimate', 'rentZestimate', 'latitude', 'longitude', 'yearBuilt', 'livingAreaValue', 'lotAreaValue', 'propertyTaxRate', 'taxAssessedValue', 'taxAssessedYear', 'lastSoldPrice', 'zestimateMinus30', 'restimateMinus30', 'zestimateLowPercent', 'zestimateHighPercent', 'restimateLowPercent', 'restimateHighPercent', 'photoCount', 'tourViewCount']
                        
                        for key, value in row_data.items():
                            if value == '' or value is None or value == 'null':
                                row_data[key] = None
                            else:
                                # Clean up quoted strings
                                if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]  # Remove outer quotes
                                
                                if key in numeric_fields:
                                    try:
                                        # Handle numeric conversion
                                        if value and value != 'null':
                                            row_data[key] = float(value) if '.' in str(value) else int(value)
                                        else:
                                            row_data[key] = None
                                    except (ValueError, TypeError):
                                        row_data[key] = None
                                # Handle boolean fields
                                elif key in ['isListingClaimedByCurrentSignedInUser', 'isCurrentSignedInAgentResponsible', 'isCurrentSignedInUserVerifiedOwner', 'hasBadGeocode', 'isUndisclosedAddress', 'hideZestimate', 'isPremierBuilder', 'isZillowOwned', 'hasPublicVideo', 'hasApprovedThirdPartyVirtualTourUrl', 'isNonOwnerOccupied', 'isFeatured', 'isHousingConnector', 'isRentalsLeadCapMet', 'isOffMarket', 'is_showcased', 'is_listed_by_management_company']:
                                    if value and str(value).lower() in ['true', 'false']:
                                        row_data[key] = str(value).lower() == 'true'
                                    else:
                                        row_data[key] = None
                                # Handle date fields
                                elif key in ['dateSoldString']:
                                    if value and value != 'null':
                                        # Try to parse the date, if it fails keep as string
                                        try:
                                            from datetime import datetime
                                            # Handle various date formats
                                            if isinstance(value, str):
                                                # Remove quotes if present
                                                clean_value = value.strip('"')
                                                # Check if it looks like a date (contains dashes and numbers)
                                                if '-' in clean_value and any(c.isdigit() for c in clean_value):
                                                    # Try common date formats
                                                    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                                                        try:
                                                            datetime.strptime(clean_value, fmt)
                                                            row_data[key] = clean_value
                                                            break
                                                        except ValueError:
                                                            continue
                                                    else:
                                                        row_data[key] = None  # Not a valid date
                                                else:
                                                    row_data[key] = None  # Doesn't look like a date
                                            else:
                                                row_data[key] = None
                                        except:
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle structured JSON fields
                                elif key in ['address']:
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the JSON string
                                            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                                                parsed_address = json.loads(value)
                                                # Extract individual address components
                                                city = parsed_address.get('city')
                                                state = parsed_address.get('state')
                                                street_address = parsed_address.get('streetAddress')
                                                zipcode = parsed_address.get('zipcode')
                                                
                                                row_data['city'] = city
                                                row_data['state'] = state
                                                row_data['streetAddress'] = street_address
                                                row_data['zipcode'] = zipcode
                                                
                                                # Create combined address field
                                                address_parts = []
                                                if street_address:
                                                    address_parts.append(street_address)
                                                if city:
                                                    address_parts.append(city)
                                                if state:
                                                    address_parts.append(state)
                                                if zipcode:
                                                    address_parts.append(zipcode)
                                                
                                                row_data[key] = ', '.join(address_parts) if address_parts else None
                                            else:
                                                row_data[key] = value
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = value
                                    else:
                                        row_data[key] = None
                                # Handle photos field - extract just URLs
                                elif key == 'photos':
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the photos JSON and extract URLs
                                            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                                                photos_data = json.loads(value)
                                                photo_urls = []
                                                for photo in photos_data:
                                                    if isinstance(photo, dict) and 'mixedSources' in photo:
                                                        jpeg_sources = photo['mixedSources'].get('jpeg', [])
                                                        if jpeg_sources and len(jpeg_sources) > 0:
                                                            # Take the first (smallest) URL for each photo
                                                            photo_urls.append(jpeg_sources[0]['url'])
                                                row_data[key] = photo_urls if photo_urls else None
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError, KeyError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle HOA details as nested document
                                elif key == 'hoa_details':
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the HOA details JSON
                                            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                                                hoa_data = json.loads(value)
                                                # Create nested structure
                                                row_data[key] = {
                                                    'has_hoa': hoa_data.get('has_hoa'),
                                                    'hoa_fee_value': hoa_data.get('hoa_fee_value'),
                                                    'hoa_fee_currency': hoa_data.get('hoa_fee_currency'),
                                                    'hoa_fee_period': hoa_data.get('hoa_fee_period'),
                                                    'amenities_included': hoa_data.get('amenities_included'),
                                                    'services_included': hoa_data.get('services_included')
                                                }
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle interior_full as nested document
                                elif key == 'interior_full':
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the interior_full JSON
                                            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                                                interior_data = json.loads(value)
                                                # Create nested structure with categories
                                                interior_structured = {}
                                                for item in interior_data:
                                                    if isinstance(item, dict) and 'title' in item and 'values' in item:
                                                        title = item['title'].lower().replace(' ', '_').replace('&', 'and')
                                                        interior_structured[title] = item['values']
                                                row_data[key] = interior_structured if interior_structured else None
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle property as nested document
                                elif key == 'property':
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the property JSON
                                            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                                                property_data = json.loads(value)
                                                # Create nested structure with categories
                                                property_structured = {}
                                                for item in property_data:
                                                    if isinstance(item, dict) and 'title' in item and 'values' in item:
                                                        title = item['title'].lower().replace(' ', '_')
                                                        property_structured[title] = item['values']
                                                row_data[key] = property_structured if property_structured else None
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle construction as nested document
                                elif key == 'construction':
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the construction JSON
                                            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                                                construction_data = json.loads(value)
                                                # Create nested structure with categories
                                                construction_structured = {}
                                                for item in construction_data:
                                                    if isinstance(item, dict) and 'title' in item and 'values' in item:
                                                        title = item['title'].lower().replace(' ', '_').replace('&', 'and')
                                                        construction_structured[title] = item['values']
                                                row_data[key] = construction_structured if construction_structured else None
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Handle structured JSON fields as flattened
                                elif key in ['priceHistory', 'schools', 'taxHistory', 'interior', 'listing_provided_by']:
                                    if value and value != 'null':
                                        try:
                                            import json
                                            # Try to parse the JSON and keep as structured data
                                            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                                parsed_data = json.loads(value)
                                                row_data[key] = parsed_data
                                            else:
                                                row_data[key] = None
                                        except (json.JSONDecodeError, TypeError):
                                            row_data[key] = None
                                    else:
                                        row_data[key] = None
                                # Keep other JSON fields as strings for now
                                elif key in ['overview']:
                                    row_data[key] = value if value and value != 'null' else None
                                # Skip fields - not needed
                                elif key in ['nearbyHomes', 'nearbyNeighborhoods', 'nearbyZipcodes', 'nearbyCities', 'homeValuation']:
                                    row_data[key] = None
                                # Handle keyword fields that should be simple strings
                                elif key in ['homeType', 'currency', 'country', 'lotAreaUnits', 'livingAreaUnitsShort', 'livingAreaUnits', 'listingDataSource', 'homeStatus']:
                                    # For keyword fields, only keep simple string values
                                    if value and value != 'null' and not value.startswith('[') and not value.startswith('{'):
                                        row_data[key] = value
                                    else:
                                        row_data[key] = None
                                # Clean up other string fields
                                else:
                                    row_data[key] = value if value and value != 'null' else None
                        
                        # Validate the record before adding it
                        # Skip records that have too many None values (likely parsing errors)
                        none_count = sum(1 for v in row_data.values() if v is None)
                        total_fields = len(row_data)
                        
                        # If more than 80% of fields are None, skip this record
                        if none_count / total_fields > 0.8:
                            logger.warning(f"Skipping record {row_count} - too many parsing errors ({none_count}/{total_fields} fields are None)")
                            continue
                        
                        data.append(row_data)
                    
                    # Log progress every 100 records
                    if row_count % 100 == 0:
                        logger.info(f"Processed {row_count} CSV rows, loaded {len(data)} valid records...")
                        
                except Exception as e:
                    logger.warning(f"Error processing row {row_count}: {e}")
                    continue
                    
        logger.info(f"Successfully loaded {len(data)} records from {row_count} CSV rows")

    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise
    
    def generate_actions():
        """Generator function for bulk actions."""
        for i, record in enumerate(data):
            # Create location field from latitude and longitude if both exist
            if 'latitude' in record and 'longitude' in record:
                if record['latitude'] is not None and record['longitude'] is not None:
                    record['location'] = {
                        "lat": float(record['latitude']),
                        "lon": float(record['longitude'])
                    }
            
            yield {
                "_index": INDEX_NAME,
                "_source": record
            }
    
    # Use parallel bulk helpers for ingestion
    success_count = 0
    error_count = 0
    
    try:
        for success, info in helpers.parallel_bulk(
            es,
            generate_actions(),
            chunk_size=CHUNK_SIZE,
            thread_count=THREAD_COUNT,
            request_timeout=60
        ):
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Failed to index document: {info}")
            
            # Log progress every 1000 records
            if (success_count + error_count) % 1000 == 0:
                logger.info(f"Processed {success_count + error_count} records...")
    
    except Exception as e:
        logger.error(f"Error during bulk ingestion: {e}")
        if hasattr(e, 'errors') and e.errors:
            logger.error(f"First few errors: {e.errors[:3]}")
        raise
    
    logger.info(f"Ingestion completed. Success: {success_count}, Errors: {error_count}")
    
    # Refresh the index to make data searchable
    es.indices.refresh(index=INDEX_NAME)
    logger.info("Index refreshed")


def main():
    """Main function to orchestrate the data ingestion process."""
    try:
        logger.info("Starting Zillow Elasticsearch data ingestion")
        
        # Validate environment
        validate_environment()
        
        # Download and extract data
        download_and_extract_data()
        
        # Create Elasticsearch client
        es = create_elasticsearch_client()
        
        # Create index
        create_index(es)
        
        # Create search template
        create_search_template(es)
        
        # Ingest data
        ingest_data(es)
        
        # Test search template
        test_search_template(es)
        
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


def create_template_only():
    """Standalone function to create only the search template."""
    try:
        logger.info("Creating search template only")
        
        # Validate environment
        validate_environment()
        
        # Create Elasticsearch client
        es = create_elasticsearch_client()
        
        # Create search template
        create_search_template(es)
        
        # Test search template
        test_search_template(es)
        
        logger.info("Search template creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Search template creation failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Check if user wants to create template only
    if len(sys.argv) > 1 and sys.argv[1] == "--template-only":
        create_template_only()
    else:
        main()
