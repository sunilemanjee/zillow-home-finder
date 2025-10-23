"""
Elasticsearch search service for executing home searches using search templates.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from elasticsearch import Elasticsearch
from dataclasses import dataclass

# OpenTelemetry imports
from opentelemetry import trace

logger = logging.getLogger(__name__)

# Get tracer from the global context
tracer = trace.get_tracer(__name__)

@dataclass
class SearchResult:
    """Represents a single home search result."""
    address: str
    price: Optional[float]
    bedrooms: Optional[int]
    bathrooms: Optional[float]
    sqft: Optional[int]
    tax_assessed_value: Optional[float]
    hoa_fee: Optional[float]
    features: List[str]
    score: float
    url: Optional[str] = None
    photo: Optional[str] = None

class ElasticsearchSearchService:
    """Service for searching homes in Elasticsearch using search templates."""
    
    def __init__(self, es_url: str, es_api_key: str, index_name: str, search_template_name: str):
        """
        Initialize the Elasticsearch search service.
        
        Args:
            es_url: Elasticsearch URL
            es_api_key: Elasticsearch API key
            index_name: Name of the Elasticsearch index
            search_template_name: Name of the search template to use
        """
        self.index_name = index_name
        self.search_template_name = search_template_name
        
        # Initialize Elasticsearch client
        self.es = Elasticsearch(
            es_url,
            api_key=es_api_key,
            request_timeout=30
        )
        
        logger.info(f"Initialized Elasticsearch client for index: {index_name}")
    
    def search_homes(self, search_params: Dict[str, Any]) -> List[SearchResult]:
        """
        Search for homes using the search template with provided parameters.
        
        Args:
            search_params: Dictionary of search parameters matching the template
            
        Returns:
            List of SearchResult objects
        """
        # Create span for search execution
        with tracer.start_as_current_span("search_service.search_homes") as span:
            span.set_attribute("search_service.index_name", self.index_name)
            span.set_attribute("search_service.template_name", self.search_template_name)
            span.set_attribute("search_service.search_params", str(search_params)[:1000])  # Truncate long params
            span.set_attribute("search_service.search_params_count", len(search_params))
            
            start_time = time.time()
            
            try:
                # Filter out None values from search parameters
                filtered_params = {k: v for k, v in search_params.items() if v is not None}
                
                # Format distance parameter for Elasticsearch (convert to "Xmi" or "Xkm" format)
                if 'distance' in filtered_params and isinstance(filtered_params['distance'], (int, float)):
                    distance_value = filtered_params['distance']
                    # Check if distance_unit is specified, default to miles
                    distance_unit = filtered_params.get('distance_unit', 'mi')
                    if distance_unit not in ['mi', 'km', 'miles', 'kilometers']:
                        distance_unit = 'mi'  # Default to miles if invalid unit
                    elif distance_unit in ['miles', 'kilometers']:
                        distance_unit = 'mi' if distance_unit == 'miles' else 'km'
                    
                    filtered_params['distance'] = f"{distance_value}{distance_unit}"
                    # Remove distance_unit from params as it's not needed by Elasticsearch
                    filtered_params.pop('distance_unit', None)
                    logger.info(f"Formatted distance parameter: {distance_value} {distance_unit} -> {filtered_params['distance']}")
                    span.set_attribute("search_service.distance_formatted", filtered_params['distance'])
                
                span.set_attribute("search_service.filtered_params_count", len(filtered_params))
                span.set_attribute("search_service.filtered_params", str(filtered_params)[:1000])  # Truncate long params
                
                logger.info(f"Searching homes with parameters: {filtered_params}")
                
                # Execute search using the template
                response = self.es.search_template(
                    index=self.index_name,
                    id=self.search_template_name,
                    params=filtered_params
                )
                
                # Record Elasticsearch response metadata
                total_hits = response.get('hits', {}).get('total', {})
                if isinstance(total_hits, dict):
                    total_count = total_hits.get('value', 0)
                else:
                    total_count = total_hits
                
                span.set_attribute("search_service.es_total_hits", total_count)
                span.set_attribute("search_service.es_took_ms", response.get('took', 0))
                
                # Parse results
                hits = response.get('hits', {}).get('hits', [])
                results = []
                
                logger.info(f"Processing {len(hits)} search hits")
                span.set_attribute("search_service.hits_to_process", len(hits))
                
                for hit in hits:
                    try:
                        result = self._parse_search_hit(hit)
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to parse search hit: {e}")
                        span.add_event("search_hit_parse_error", {"error": str(e)})
                        continue
                
                logger.info(f"Found {len(results)} home search results")
                
                # Record search results
                span.set_attribute("search_service.success", True)
                span.set_attribute("search_service.results_count", len(results))
                span.set_attribute("search_service.results_parsed", len(results))
                
                # Record execution time
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                span.set_attribute("search_service.execution_time_ms", execution_time)
                
                return results
                
            except Exception as e:
                logger.error(f"Error during home search: {e}")
                span.record_exception(e)
                span.set_attribute("search_service.success", False)
                span.set_attribute("search_service.error", str(e))
                return []
    
    def _parse_search_hit(self, hit: Dict[str, Any]) -> Optional[SearchResult]:
        """
        Parse a single search hit into a SearchResult object.
        
        Args:
            hit: Single search hit from Elasticsearch
            
        Returns:
            SearchResult object or None if parsing fails
        """
        try:
            fields = hit.get('fields', {})
            
            # Extract address
            address_list = fields.get('address', [])
            address = address_list[0] if address_list else "Address not available"
            
            # Extract price
            price_list = fields.get('price', [])
            price = float(price_list[0]) if price_list and price_list[0] is not None else None
            
            # Extract bedrooms
            bedrooms_list = fields.get('bedrooms', [])
            bedrooms = int(bedrooms_list[0]) if bedrooms_list and bedrooms_list[0] is not None else None
            
            # Extract bathrooms
            bathrooms_list = fields.get('bathrooms', [])
            bathrooms = float(bathrooms_list[0]) if bathrooms_list and bathrooms_list[0] is not None else None
            
            # Extract square footage
            sqft_list = fields.get('sqft', [])
            sqft = int(sqft_list[0]) if sqft_list and sqft_list[0] is not None else None
            
            # Extract tax assessed value
            tax_list = fields.get('taxAssessedValue', [])
            tax_assessed_value = float(tax_list[0]) if tax_list and tax_list[0] is not None else None
            
            # Extract HOA fee
            hoa_list = fields.get('hoa_details.hoa_fee_value', [])
            hoa_fee = float(hoa_list[0]) if hoa_list and hoa_list[0] is not None else None
            
            # Extract features
            features = []
            property_features = fields.get('property.features', [])
            interior_features = fields.get('interior_full.features', [])
            
            if property_features:
                features.extend(property_features)
            if interior_features:
                features.extend(interior_features)
            
            # Extract URL
            url_list = fields.get('url', [])
            url = url_list[0] if url_list and url_list[0] else None
            if url:
                # Ensure URL is properly formatted as a string
                url = str(url).strip()
            
            # Extract photo
            photo_list = fields.get('photos[0]', [])
            photo = photo_list[0] if photo_list and photo_list[0] else None
            
            # Get relevance score
            score = hit.get('_score', 0.0)
            
            return SearchResult(
                address=address,
                price=price,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft=sqft,
                tax_assessed_value=tax_assessed_value,
                hoa_fee=hoa_fee,
                features=features,
                score=score,
                url=url,
                photo=photo
            )
            
        except Exception as e:
            logger.error(f"Error parsing search hit: {e}")
            return None
    
    def format_results_for_display(self, results: List[SearchResult]) -> str:
        """
        Format search results for display to the user.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string representation of results with markdown formatting
        """
        if not results:
            return "No homes found matching your criteria."
        
        formatted_results = []
        formatted_results.append(f"## Found {len(results)} homes\n")
        
        for i, result in enumerate(results, 1):
            # Create address as hyperlink if URL is available
            if result.url:
                # Ensure the URL is properly formatted
                clean_url = result.url.strip()
                markdown_link = f"### {i}. [{result.address}]({clean_url})\n"
                # Add the full URL as text below the link for reference
                markdown_link += f"🔗 **Full URL:** `{clean_url}`\n"
                result_text = markdown_link
            else:
                result_text = f"### {i}. {result.address}\n"
            
            # Add photo if available (place it prominently)
            if result.photo:
                result_text += f"![Property Photo]({result.photo})\n\n"
            
            # Property details
            details = []
            if result.price:
                details.append(f"**Price:** ${result.price:,.0f}")
            
            if result.bedrooms:
                details.append(f"**Bedrooms:** {result.bedrooms}")
            
            if result.bathrooms:
                details.append(f"**Bathrooms:** {result.bathrooms}")
            
            if result.sqft:
                details.append(f"**Square Feet:** {result.sqft:,}")
            
            if result.tax_assessed_value:
                details.append(f"**Tax Assessed Value:** ${result.tax_assessed_value:,.0f}")
            
            if result.hoa_fee:
                details.append(f"**HOA Fee:** ${result.hoa_fee:.2f}/month")
            
            if details:
                result_text += " | ".join(details) + "\n\n"
            
            if result.features:
                result_text += f"**Features:** {', '.join(result.features[:5])}\n"  # Limit to first 5 features
            
            result_text += f"\n*Relevance Score: {result.score:.2f}*\n"
            
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
