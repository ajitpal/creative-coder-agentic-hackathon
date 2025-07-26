"""
WHO ICD API client for medical term and disease definition lookup
Provides authoritative medical terminology from World Health Organization
"""

import requests
import asyncio
from typing import Dict, Any, List, Optional
import json
from urllib.parse import quote
import time
from src.config import Config
from src.models import WHOICDResponse
from src.utils import setup_logging

logger = setup_logging()

class WHOICDClient:
    """Client for WHO ICD-11 API for medical term definitions"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize WHO ICD API client"""
        self.base_url = base_url or "https://id.who.int"
        
        # OAuth2 credentials
        self.client_id = Config.WHO_ICD_CLIENT_ID
        self.client_secret = Config.WHO_ICD_CLIENT_SECRET
        self.token_url = Config.WHO_ICD_TOKEN_URL
        
        # ICD-11 configuration
        self.release_id = "2019-04"  # Latest stable release
        self.linearization = "mms"   # Mortality and Morbidity Statistics
        
        # API endpoints based on swagger documentation
        self.search_endpoint = f"{self.base_url}/icd/release/11/{self.release_id}/{self.linearization}/search"
        self.entity_endpoint = f"{self.base_url}/icd/release/11/{self.release_id}/{self.linearization}"
        
        # OAuth2 token management
        self.access_token = None
        self.token_expires_at = 0
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Accept-Language': 'en',
            'API-Version': 'v2'
        })
        
        logger.info("WHO ICD API client initialized")
    
    async def _get_access_token(self) -> str:
        """Get OAuth2 access token using client credentials flow"""
        current_time = time.time()
        
        # Check if we have a valid token
        if self.access_token and current_time < self.token_expires_at:
            return self.access_token
        
        try:
            # Request new token
            token_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'icdapi_access'
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(self.token_url, data=token_data, headers=headers, timeout=10)
            )
            
            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response['access_token']
                expires_in = token_response.get('expires_in', 3600)  # Default 1 hour
                self.token_expires_at = current_time + expires_in - 60  # Refresh 1 minute early
                
                logger.info("WHO ICD OAuth2 token obtained successfully")
                return self.access_token
            else:
                logger.error(f"Failed to get WHO ICD OAuth2 token: {response.status_code} - {response.text}")
                raise Exception(f"OAuth2 token request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting WHO ICD OAuth2 token: {e}")
            raise Exception(f"OAuth2 authentication failed: {str(e)}")
    
    async def _make_authenticated_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an authenticated request to the WHO ICD API"""
        try:
            # Get access token
            access_token = await self._get_access_token()
            
            # Add authorization header
            headers = kwargs.get('headers', {})
            headers['Authorization'] = f'Bearer {access_token}'
            kwargs['headers'] = headers
            
            # Make the request
            loop = asyncio.get_event_loop()
            if method.upper() == 'GET':
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(url, **kwargs)
                )
            elif method.upper() == 'POST':
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.post(url, **kwargs)
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error making authenticated request: {e}")
            raise
    
    async def search_term(
        self, 
        term: str, 
        max_results: int = 10,
        use_fuzzy_search: bool = True
    ) -> Dict[str, Any]:
        """Search for medical term in WHO ICD database"""
        try:
            logger.info(f"Searching WHO ICD for term: {term}")
            
            # Prepare search parameters
            params = {
                'q': term,
                'subtreeFilterUsesFoundationDescendants': 'false',
                'includeKeywordResult': 'true',
                'useFlexisearch': str(use_fuzzy_search).lower(),
                'flatResults': 'false',
                'highlightingEnabled': 'false'
            }
            
            # Make authenticated async request
            response = await self._make_authenticated_request(
                'GET', 
                self.search_endpoint, 
                params=params, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WHO ICD response summary for '{term}': {len(data.get('destinationEntities', []))} entities found")
                return await self._process_search_results(data, term, max_results)
            
            elif response.status_code == 401:
                logger.error("WHO ICD API authentication failed")
                return {
                    'term': term,
                    'success': False,
                    'error': 'Authentication failed - check API key',
                    'results': []
                }
            
            elif response.status_code == 429:
                logger.warning("WHO ICD API rate limit exceeded")
                return {
                    'term': term,
                    'success': False,
                    'error': 'Rate limit exceeded - try again later',
                    'results': []
                }
            
            else:
                logger.error(f"WHO ICD API error: {response.status_code}")
                return {
                    'term': term,
                    'success': False,
                    'error': f'API error: {response.status_code}',
                    'results': []
                }
                
        except requests.exceptions.Timeout:
            logger.error("WHO ICD API request timeout")
            return {
                'term': term,
                'success': False,
                'error': 'Request timeout',
                'results': []
            }
        
        except requests.exceptions.ConnectionError:
            logger.error("WHO ICD API connection error")
            return {
                'term': term,
                'success': False,
                'error': 'Connection error',
                'results': []
            }
        
        except Exception as e:
            logger.error(f"WHO ICD API unexpected error: {e}")
            return {
                'term': term,
                'success': False,
                'error': str(e),
                'results': []
            }
    
    async def _process_search_results(
        self, 
        data: Dict[str, Any], 
        original_term: str, 
        max_results: int
    ) -> Dict[str, Any]:
        """Process and format search results from WHO ICD API"""
        try:
            results = []
            
            # Extract destination entities (main results)
            destinations = data.get('destinationEntities', [])
            
            for i, entity in enumerate(destinations[:max_results]):
                # Create WHOICDResponse from entity data
                who_response = WHOICDResponse.from_api_response(entity)
                
                if who_response.is_valid():
                    # Get additional details if available
                    entity_details = await self._get_entity_details(entity.get('@id', ''))
                    
                    result = {
                        'rank': i + 1,
                        'title': who_response.title,
                        'definition': who_response.definition,
                        'code': who_response.code,
                        'synonyms': who_response.synonyms,
                        'url': entity.get('@id', ''),
                        'match_score': self._calculate_match_score(original_term, who_response.title),
                        'details': entity_details
                    }
                    
                    results.append(result)
            
            # Get best match (highest score)
            best_match = None
            if results:
                best_match = max(results, key=lambda x: x['match_score'])
            
            return {
                'term': original_term,
                'success': True,
                'total_results': len(destinations),
                'returned_results': len(results),
                'best_match': best_match,
                'results': results,
                'definition': best_match['definition'] if best_match else '',
                'code': best_match['code'] if best_match else '',
                'synonyms': best_match['synonyms'] if best_match else []
            }
            
        except Exception as e:
            logger.error(f"Error processing WHO ICD search results: {e}")
            return {
                'term': original_term,
                'success': False,
                'error': f'Error processing results: {str(e)}',
                'results': []
            }
    
    async def _get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific entity"""
        if not entity_id:
            return {}
        
        try:
            # Extract entity ID from URL if needed
            if entity_id.startswith('http'):
                entity_id = entity_id.split('/')[-1]
            
            detail_url = f"{self.entity_endpoint}/{entity_id}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get(detail_url, timeout=5)
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'parent_categories': self._extract_parent_categories(data),
                    'children': self._extract_children(data),
                    'additional_info': self._extract_additional_info(data)
                }
            
        except Exception as e:
            logger.debug(f"Could not get entity details for {entity_id}: {e}")
        
        return {}
    
    def _extract_parent_categories(self, entity_data: Dict[str, Any]) -> List[str]:
        """Extract parent categories from entity data"""
        parents = []
        try:
            parent_data = entity_data.get('parent', [])
            if isinstance(parent_data, list):
                for parent in parent_data:
                    if isinstance(parent, dict) and 'title' in parent:
                        title = parent['title']
                        if isinstance(title, dict):
                            parents.append(title.get('@value', ''))
                        else:
                            parents.append(str(title))
        except Exception as e:
            logger.debug(f"Error extracting parent categories: {e}")
        
        return parents
    
    def _extract_children(self, entity_data: Dict[str, Any]) -> List[str]:
        """Extract child categories from entity data"""
        children = []
        try:
            child_data = entity_data.get('child', [])
            if isinstance(child_data, list):
                for child in child_data[:5]:  # Limit to first 5 children
                    if isinstance(child, dict) and 'title' in child:
                        title = child['title']
                        if isinstance(title, dict):
                            children.append(title.get('@value', ''))
                        else:
                            children.append(str(title))
        except Exception as e:
            logger.debug(f"Error extracting children: {e}")
        
        return children
    
    def _extract_additional_info(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional information from entity data"""
        info = {}
        try:
            # Extract inclusion/exclusion notes
            if 'inclusion' in entity_data:
                info['inclusions'] = self._extract_text_array(entity_data['inclusion'])
            
            if 'exclusion' in entity_data:
                info['exclusions'] = self._extract_text_array(entity_data['exclusion'])
            
            # Extract coding notes
            if 'codingNote' in entity_data:
                info['coding_notes'] = self._extract_text_array(entity_data['codingNote'])
            
        except Exception as e:
            logger.debug(f"Error extracting additional info: {e}")
        
        return info
    
    def _extract_text_array(self, data: Any) -> List[str]:
        """Extract text array from various data formats"""
        texts = []
        try:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and '@value' in item:
                        texts.append(item['@value'])
                    elif isinstance(item, str):
                        texts.append(item)
            elif isinstance(data, dict) and '@value' in data:
                texts.append(data['@value'])
            elif isinstance(data, str):
                texts.append(data)
        except Exception:
            pass
        
        return texts
    
    def _calculate_match_score(self, search_term: str, result_title: str) -> float:
        """Calculate relevance score for search results"""
        if not search_term or not result_title:
            return 0.0
        
        search_lower = search_term.lower().strip()
        title_lower = result_title.lower().strip()
        
        # Exact match gets highest score
        if search_lower == title_lower:
            return 1.0
        
        # Check if search term is contained in title
        if search_lower in title_lower:
            return 0.8
        
        # Check if title starts with search term
        if title_lower.startswith(search_lower):
            return 0.9
        
        # Word-based matching
        search_words = set(search_lower.split())
        title_words = set(title_lower.split())
        
        if search_words:
            common_words = search_words.intersection(title_words)
            word_score = len(common_words) / len(search_words)
            return word_score * 0.7
        
        return 0.0
    
    async def get_disease_hierarchy(self, disease_code: str) -> Dict[str, Any]:
        """Get disease hierarchy information"""
        try:
            logger.info(f"Getting hierarchy for disease code: {disease_code}")
            
            entity_url = f"{self.entity_endpoint}/{disease_code}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get(entity_url, timeout=10)
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'code': disease_code,
                    'success': True,
                    'hierarchy': {
                        'parents': self._extract_parent_categories(data),
                        'children': self._extract_children(data),
                        'level': self._determine_hierarchy_level(data)
                    },
                    'additional_info': self._extract_additional_info(data)
                }
            
            else:
                return {
                    'code': disease_code,
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"Error getting disease hierarchy: {e}")
            return {
                'code': disease_code,
                'success': False,
                'error': str(e)
            }
    
    def _determine_hierarchy_level(self, entity_data: Dict[str, Any]) -> str:
        """Determine the hierarchy level of an entity"""
        try:
            # This is a simplified determination - actual ICD hierarchy is complex
            if 'parent' not in entity_data or not entity_data['parent']:
                return 'chapter'  # Top level
            elif 'child' in entity_data and entity_data['child']:
                return 'category'  # Has children
            else:
                return 'subcategory'  # Leaf node
        except Exception:
            return 'unknown'
    
    async def validate_medical_code(self, code: str) -> Dict[str, Any]:
        """Validate if a medical code exists in WHO ICD"""
        try:
            logger.info(f"Validating medical code: {code}")
            
            entity_url = f"{self.entity_endpoint}/{code}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get(entity_url, timeout=5)
            )
            
            if response.status_code == 200:
                data = response.json()
                who_response = WHOICDResponse.from_api_response(data)
                
                return {
                    'code': code,
                    'valid': True,
                    'title': who_response.title,
                    'definition': who_response.definition
                }
            
            elif response.status_code == 404:
                return {
                    'code': code,
                    'valid': False,
                    'error': 'Code not found'
                }
            
            else:
                return {
                    'code': code,
                    'valid': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"Error validating medical code: {e}")
            return {
                'code': code,
                'valid': False,
                'error': str(e)
            }
    
    def test_connection(self) -> bool:
        """Test connection to WHO ICD API"""
        try:
            # Test with a simple search
            response = self.session.get(
                self.search_endpoint,
                params={'q': 'diabetes', 'flatResults': 'true'},
                timeout=5
            )
            
            success = response.status_code in [200, 401]  # 401 means API is reachable but auth failed
            logger.info(f"WHO ICD API connection test: {'PASSED' if success else 'FAILED'}")
            return success
            
        except Exception as e:
            logger.error(f"WHO ICD API connection test failed: {e}")
            return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client configuration information"""
        return {
            'base_url': self.base_url,
            'client_id_configured': bool(self.client_id),
            'client_secret_configured': bool(self.client_secret),
            'endpoints': {
                'search': self.search_endpoint,
                'entity': self.entity_endpoint,
                'token': self.token_url
            }
        }
    
    async def search_medical_term(self, term: str) -> Dict[str, Any]:
        """Search for medical term - alias for search_term method"""
        return await self.search_term(term)
    
    def close(self):
        """Close the HTTP session"""
        if self.session:
            self.session.close()
        logger.info("WHO ICD API client closed")