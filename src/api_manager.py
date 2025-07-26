"""
Unified API Manager for Intelligent Healthcare Navigator
Coordinates all external API calls with error handling, retry logic, and fallback mechanisms
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
import time
from dataclasses import dataclass
from enum import Enum
import json

from src.gemini_client import GeminiAPIClient
from src.who_icd_client import WHOICDClient
from src.openfda_client import OpenFDAClient
from src.memory import CacheManager
from src.config import Config
from src.utils import setup_logging

logger = setup_logging()

class APIStatus(Enum):
    """API status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class APIResult:
    """Standardized API result container"""
    success: bool
    data: Dict[str, Any]
    source: str
    response_time: float
    cached: bool = False
    error: Optional[str] = None
    retry_count: int = 0

class APIManager:
    """Unified manager for all external API integrations"""
    
    def __init__(self, cache_manager: CacheManager = None):
        """Initialize API manager with all clients"""
        self.cache_manager = cache_manager or CacheManager()
        
        # Initialize API clients
        self.gemini_client: Optional[GeminiAPIClient] = None
        self.who_icd_client: Optional[WHOICDClient] = None
        self.openfda_client: Optional[OpenFDAClient] = None
        
        # API status tracking
        self.api_status: Dict[str, APIStatus] = {}
        self.last_status_check: Dict[str, float] = {}
        self.status_check_interval = 300  # 5 minutes
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delays = [1, 2, 4]  # Exponential backoff
        
        # Initialize clients
        self._initialize_clients()
        
        logger.info("API Manager initialized")
    
    def _initialize_clients(self):
        """Initialize all API clients with error handling"""
        try:
            # Initialize Gemini client
            if Config.GEMINI_API_KEY:
                self.gemini_client = GeminiAPIClient()
                self.api_status['gemini'] = APIStatus.UNKNOWN
                logger.info("Gemini API client initialized")
            else:
                logger.warning("Gemini API key not configured")
                self.api_status['gemini'] = APIStatus.UNAVAILABLE
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.api_status['gemini'] = APIStatus.ERROR
        
        try:
            # Initialize WHO ICD client
            self.who_icd_client = WHOICDClient()
            self.api_status['who_icd'] = APIStatus.UNKNOWN
            logger.info("WHO ICD API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WHO ICD client: {e}")
            self.api_status['who_icd'] = APIStatus.ERROR
        
        try:
            # Initialize OpenFDA client
            self.openfda_client = OpenFDAClient()
            self.api_status['openfda'] = APIStatus.UNKNOWN
            logger.info("OpenFDA API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenFDA client: {e}")
            self.api_status['openfda'] = APIStatus.ERROR
    
    async def check_api_status(self, api_name: str = None) -> Dict[str, APIStatus]:
        """Check status of APIs"""
        apis_to_check = [api_name] if api_name else ['gemini', 'who_icd', 'openfda']
        
        for api in apis_to_check:
            current_time = time.time()
            last_check = self.last_status_check.get(api, 0)
            
            # Skip if recently checked
            if current_time - last_check < self.status_check_interval:
                continue
            
            try:
                if api == 'gemini' and self.gemini_client:
                    success = self.gemini_client.test_connection()
                    self.api_status[api] = APIStatus.AVAILABLE if success else APIStatus.ERROR
                
                elif api == 'who_icd' and self.who_icd_client:
                    success = self.who_icd_client.test_connection()
                    self.api_status[api] = APIStatus.AVAILABLE if success else APIStatus.ERROR
                
                elif api == 'openfda' and self.openfda_client:
                    success = self.openfda_client.test_connection()
                    self.api_status[api] = APIStatus.AVAILABLE if success else APIStatus.ERROR
                
                self.last_status_check[api] = current_time
                logger.debug(f"API status check for {api}: {self.api_status[api].value}")
                
            except Exception as e:
                logger.error(f"Error checking {api} API status: {e}")
                self.api_status[api] = APIStatus.ERROR
        
        return self.api_status
    
    async def _execute_with_retry(
        self, 
        func, 
        api_name: str, 
        cache_key: str = None,
        *args, 
        **kwargs
    ) -> APIResult:
        """Execute API call with retry logic and caching"""
        start_time = time.time()
        
        # Check cache first
        if cache_key:
            cached_result = self.cache_manager.get_cached_response(api_name, "generic", {"key": cache_key})
            if cached_result:
                return APIResult(
                    success=True,
                    data=cached_result,
                    source=api_name,
                    response_time=time.time() - start_time,
                    cached=True
                )
        
        # Check API status
        await self.check_api_status(api_name)
        if self.api_status.get(api_name) == APIStatus.UNAVAILABLE:
            return APIResult(
                success=False,
                data={},
                source=api_name,
                response_time=time.time() - start_time,
                error=f"{api_name} API is unavailable"
            )
        
        # Execute with retry logic
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                response_time = time.time() - start_time
                
                # Handle different result types
                if hasattr(result, 'text'):  # GeminiResponse object
                    success = bool(result.text)
                    result_data = result
                elif isinstance(result, dict):
                    success = result.get('success', True)
                    result_data = result
                else:
                    success = bool(result)
                    result_data = result
                
                # Cache successful results
                if cache_key and success:
                    self.cache_manager.cache_api_response(
                        api_name, 
                        "generic", 
                        {"key": cache_key}, 
                        result_data
                    )
                
                return APIResult(
                    success=success,
                    data=result_data,
                    source=api_name,
                    response_time=response_time,
                    retry_count=attempt
                )
                
            except Exception as e:
                last_exception = e
                logger.warning(f"API call attempt {attempt + 1} failed for {api_name}: {e}")
                
                # Update API status on certain errors
                if "rate limit" in str(e).lower():
                    self.api_status[api_name] = APIStatus.RATE_LIMITED
                elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                    self.api_status[api_name] = APIStatus.ERROR
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
        
        # All retries failed
        response_time = time.time() - start_time
        return APIResult(
            success=False,
            data={},
            source=api_name,
            response_time=response_time,
            error=str(last_exception),
            retry_count=self.max_retries
        )
    
    async def search_medical_term(self, term: str) -> APIResult:
        """Search for medical term using WHO ICD API with fallbacks"""
        logger.info(f"Searching medical term: {term}")
        
        cache_key = f"medical_term_{term.lower().replace(' ', '_')}"
        
        if self.who_icd_client:
            result = await self._execute_with_retry(
                self.who_icd_client.search_term,
                "who_icd",
                cache_key,
                term
            )
            
            if result.success and result.data.get('success'):
                return result
            
            logger.warning(f"WHO ICD search failed for {term}, trying fallback")
        
        # Fallback to Gemini for medical term explanation
        if self.gemini_client:
            fallback_prompt = f"""
            Please provide a clear, medically accurate explanation of the term "{term}".
            Include:
            1. Definition in plain language
            2. Common synonyms or alternative names
            3. Basic medical context
            4. Any important notes for patients
            
            Format as JSON with keys: definition, synonyms, medical_context, patient_notes
            """
            
            try:
                gemini_result = await self._execute_with_retry(
                    self.gemini_client.generate_response,
                    "gemini_fallback",
                    f"fallback_{cache_key}",
                    fallback_prompt,
                    use_functions=False
                )
                
                if gemini_result.success:
                    # Format Gemini response to match WHO ICD format
                    formatted_data = {
                        'term': term,
                        'success': True,
                        'source': 'gemini_fallback',
                        'definition': gemini_result.data.text if hasattr(gemini_result.data, 'text') else gemini_result.data.get('text', ''),
                        'fallback_used': True
                    }
                    
                    return APIResult(
                        success=True,
                        data=formatted_data,
                        source="gemini_fallback",
                        response_time=gemini_result.response_time
                    )
            
            except Exception as e:
                logger.error(f"Gemini fallback failed for medical term {term}: {e}")
        
        # No successful result from any source
        return APIResult(
            success=False,
            data={'term': term, 'error': 'All medical term lookup sources failed'},
            source="none",
            response_time=0,
            error="All sources unavailable"
        )
    
    async def get_drug_information(
        self, 
        drug_name: str, 
        include_recalls: bool = True,
        include_adverse_events: bool = True
    ) -> APIResult:
        """Get comprehensive drug information from OpenFDA"""
        logger.info(f"Getting drug information: {drug_name}")
        
        cache_key = f"drug_info_{drug_name.lower().replace(' ', '_')}"
        
        if self.openfda_client:
            result = await self._execute_with_retry(
                self.openfda_client.search_drug_comprehensive,
                "openfda",
                cache_key,
                drug_name
            )
            
            if result.success:
                return result
        
        # Fallback: Try individual FDA endpoints
        if self.openfda_client:
            try:
                recalls_task = None
                events_task = None
                
                if include_recalls:
                    recalls_task = self._execute_with_retry(
                        self.openfda_client.get_drug_recalls,
                        "openfda_recalls",
                        f"recalls_{cache_key}",
                        drug_name
                    )
                
                if include_adverse_events:
                    events_task = self._execute_with_retry(
                        self.openfda_client.get_adverse_events,
                        "openfda_events",
                        f"events_{cache_key}",
                        drug_name
                    )
                
                # Execute available tasks
                tasks = [task for task in [recalls_task, events_task] if task]
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Combine results
                    combined_data = {
                        'drug_name': drug_name,
                        'success': True,
                        'partial_results': True
                    }
                    
                    if recalls_task and len(results) > 0 and not isinstance(results[0], Exception):
                        combined_data['recalls'] = results[0].data
                    
                    if events_task and len(results) > 1 and not isinstance(results[1], Exception):
                        combined_data['adverse_events'] = results[1].data
                    elif events_task and len(results) == 1 and not isinstance(results[0], Exception):
                        combined_data['adverse_events'] = results[0].data
                    
                    return APIResult(
                        success=True,
                        data=combined_data,
                        source="openfda_partial",
                        response_time=max(r.response_time for r in results if isinstance(r, APIResult))
                    )
            
            except Exception as e:
                logger.error(f"OpenFDA fallback failed for {drug_name}: {e}")
        
        return APIResult(
            success=False,
            data={'drug_name': drug_name, 'error': 'Drug information sources unavailable'},
            source="none",
            response_time=0,
            error="All drug information sources failed"
        )
    
    async def analyze_symptoms_with_gemini(
        self, 
        symptoms: str, 
        context: Dict[str, Any] = None
    ) -> APIResult:
        """Analyze symptoms using Gemini with healthcare functions"""
        logger.info(f"Analyzing symptoms with Gemini: {symptoms[:50]}...")
        
        cache_key = f"symptoms_{hash(symptoms)}"
        
        if not self.gemini_client:
            return APIResult(
                success=False,
                data={'error': 'Gemini API not available'},
                source="none",
                response_time=0,
                error="Gemini client not initialized"
            )
        
        # Prepare context-aware prompt
        prompt = f"""
        Please analyze these symptoms and provide general health insights (not medical diagnosis):
        
        Symptoms: {symptoms}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Please provide:
        1. General insights about these symptoms
        2. Urgency level assessment (low/medium/high/emergency)
        3. General recommendations (always include seeking professional medical care)
        4. Important disclaimers
        
        Remember: Do not provide medical diagnosis. Focus on general health education and when to seek care.
        """
        
        result = await self._execute_with_retry(
            self.gemini_client.generate_response,
            "gemini",
            cache_key,
            prompt,
            context=context.get('conversation_history', []) if context else None,
            use_functions=True
        )
        
        return result
    
    async def extract_medical_entities(self, text: str) -> APIResult:
        """Extract medical entities from text using available NLP tools"""
        logger.info(f"Extracting medical entities from text: {text[:50]}...")
        
        cache_key = f"entities_{hash(text)}"
        
        # For now, use Gemini as the primary entity extraction method
        if self.gemini_client:
            prompt = f"""
            Extract medical entities from the following text. Identify:
            - Diseases and conditions
            - Medications and drugs
            - Symptoms
            - Medical procedures
            - Anatomical terms
            
            Text: {text}
            
            Return results as JSON with format:
            {{
                "entities": [
                    {{"text": "entity", "type": "disease|drug|symptom|procedure|anatomy", "confidence": 0.0-1.0}}
                ]
            }}
            """
            
            result = await self._execute_with_retry(
                self.gemini_client.generate_response,
                "gemini",
                cache_key,
                prompt,
                use_functions=False
            )
            
            if result.success:
                # Try to parse JSON from Gemini response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', result.data.get('text', ''), re.DOTALL)
                    if json_match:
                        entities_data = json.loads(json_match.group())
                        result.data['parsed_entities'] = entities_data
                except Exception as e:
                    logger.warning(f"Could not parse entities JSON: {e}")
            
            return result
        
        return APIResult(
            success=False,
            data={'error': 'No entity extraction services available'},
            source="none",
            response_time=0,
            error="Entity extraction not available"
        )
    
    async def get_comprehensive_health_info(
        self, 
        query: str, 
        query_type: str = "general",
        context: Dict[str, Any] = None
    ) -> Dict[str, APIResult]:
        """Get comprehensive health information using multiple APIs"""
        logger.info(f"Getting comprehensive health info for: {query}")
        
        results = {}
        
        # Determine what information to gather based on query type
        if query_type == "medical_term":
            results['medical_term'] = await self.search_medical_term(query)
        
        elif query_type == "drug_info":
            results['drug_info'] = await self.get_drug_information(query)
        
        elif query_type == "symptoms":
            results['symptom_analysis'] = await self.analyze_symptoms_with_gemini(query, context)
        
        elif query_type == "entity_extraction":
            results['entities'] = await self.extract_medical_entities(query)
        
        else:  # General query - try to determine what's needed
            # Use Gemini to analyze the query and determine what information to gather
            if self.gemini_client:
                analysis_result = await self._execute_with_retry(
                    self.gemini_client.analyze_query_intent,
                    "gemini",
                    f"analysis_{hash(query)}",
                    query
                )
                
                if analysis_result.success:
                    intent_data = analysis_result.data
                    suggested_functions = intent_data.get('suggested_functions', [])
                    
                    # Execute suggested functions
                    tasks = []
                    if 'search_medical_term' in suggested_functions:
                        tasks.append(('medical_term', self.search_medical_term(query)))
                    
                    if 'get_drug_information' in suggested_functions:
                        tasks.append(('drug_info', self.get_drug_information(query)))
                    
                    if 'analyze_symptoms' in suggested_functions:
                        tasks.append(('symptoms', self.analyze_symptoms_with_gemini(query, context)))
                    
                    # Execute all tasks concurrently
                    if tasks:
                        task_results = await asyncio.gather(
                            *[task[1] for task in tasks], 
                            return_exceptions=True
                        )
                        
                        for i, (result_key, _) in enumerate(tasks):
                            if i < len(task_results) and not isinstance(task_results[i], Exception):
                                results[result_key] = task_results[i]
                    
                    results['query_analysis'] = analysis_result
        
        return results
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get statistics about API usage and status"""
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            'api_status': {name: status.value for name, status in self.api_status.items()},
            'last_status_check': self.last_status_check,
            'cache_statistics': cache_stats,
            'clients_initialized': {
                'gemini': self.gemini_client is not None,
                'who_icd': self.who_icd_client is not None,
                'openfda': self.openfda_client is not None
            },
            'retry_configuration': {
                'max_retries': self.max_retries,
                'retry_delays': self.retry_delays
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all APIs"""
        logger.info("Performing API health check")
        
        health_status = {}
        
        # Check all APIs
        api_status = await self.check_api_status()
        
        for api_name, status in api_status.items():
            health_status[api_name] = {
                'status': status.value,
                'last_checked': self.last_status_check.get(api_name, 0),
                'client_available': getattr(self, f"{api_name}_client") is not None
            }
        
        # Overall system health
        available_apis = sum(1 for status in api_status.values() if status == APIStatus.AVAILABLE)
        total_apis = len(api_status)
        
        health_status['overall'] = {
            'healthy': available_apis > 0,
            'available_apis': available_apis,
            'total_apis': total_apis,
            'health_percentage': round((available_apis / total_apis) * 100, 1) if total_apis > 0 else 0
        }
        
        return health_status
    
    def close(self):
        """Close all API clients and cleanup resources"""
        try:
            if self.gemini_client:
                # Gemini client doesn't have explicit close method
                pass
            
            if self.who_icd_client:
                self.who_icd_client.close()
            
            if self.openfda_client:
                self.openfda_client.close()
            
            if self.cache_manager:
                self.cache_manager.close()
            
            logger.info("API Manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing API Manager: {e}")