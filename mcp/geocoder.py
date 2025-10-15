"""
Azure Maps geocoding service for converting location names to coordinates.
"""

import logging
import requests
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """Represents a geocoded location with coordinates and address."""
    latitude: float
    longitude: float
    address: str
    confidence: Optional[str] = None

class AzureMapsGeocoder:
    """Azure Maps geocoding service client."""
    
    def __init__(self, subscription_key: str):
        """
        Initialize the Azure Maps geocoder.
        
        Args:
            subscription_key: Azure Maps subscription key
        """
        self.subscription_key = subscription_key
        self.base_url = "https://atlas.microsoft.com/search/address/json"
    
    def geocode(self, location_query: str) -> Optional[Location]:
        """
        Geocode a location query to get latitude and longitude.
        
        Args:
            location_query: Location string to geocode (e.g., "Orlando FL", "New York City")
            
        Returns:
            Location object with coordinates and address, or None if geocoding fails
        """
        try:
            # Debug logging
            logger.info(f"Geocoding location: {location_query}")
            logger.info(f"Using subscription key: {self.subscription_key[:10] if self.subscription_key else 'None'}...")
            logger.info(f"Base URL: {self.base_url}")
            
            params = {
                'api-version': '1.0',
                'subscription-key': self.subscription_key,
                'query': location_query,
                'limit': 1
            }
            
            logger.info(f"Request params: {params}")
            response = requests.get(self.base_url, params=params, timeout=10)
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('results'):
                logger.warning(f"No results found for location: {location_query}")
                return None
            
            result = data['results'][0]
            position = result['position']
            address = result.get('address', {}).get('freeformAddress', location_query)
            confidence = result.get('score', 'unknown')
            
            location = Location(
                latitude=float(position['lat']),
                longitude=float(position['lon']),
                address=address,
                confidence=str(confidence)
            )
            
            logger.info(f"Successfully geocoded '{location_query}' to {location.latitude}, {location.longitude}")
            return location
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during geocoding: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing geocoding response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during geocoding: {e}")
            return None
    
    def is_valid_location(self, location_query: str) -> bool:
        """
        Check if a location query is valid by attempting to geocode it.
        
        Args:
            location_query: Location string to validate
            
        Returns:
            True if the location can be geocoded, False otherwise
        """
        return self.geocode(location_query) is not None
