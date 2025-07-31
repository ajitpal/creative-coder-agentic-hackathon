"""
OpenFDA API client for drug recall and adverse event information
Provides real-time drug safety data from FDA databases
"""

import requests
import asyncio
from typing import Dict, Any, List, Optional
import json
from urllib.parse import quote
from datetime import datetime, timedelta
from src.config import Config
from src.models import OpenFDAResponse
from src.utils import setup_logging

logger = setup_logging()

class OpenFDAClient:
    """Client for OpenFDA API for drug recalls and adverse events"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize OpenFDA API client"""
        self.api_key = api_key or Config.OPENFDA_API_KEY
        self.base_url = base_url or Config.OPENFDA_BASE_URL
        
        # API endpoints
        self.drug_recall_endpoint = f"{self.base_url}/drug/enforcement.json"
        self.drug_event_endpoint = f"{self.base_url}/drug/event.json"
        self.drug_label_endpoint = f"{self.base_url}/drug/label.json"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'HealthcareNavigator/1.0'
        })
        
        # Rate limiting (FDA allows 240 requests per minute for API key users, 40 without)
        self.rate_limit = 240 if self.api_key else 40
        self.request_interval = 60.0 / self.rate_limit  # Seconds between requests
        self.last_request_time = 0
        
        logger.info(f"OpenFDA API client initialized (rate limit: {self.rate_limit}/min)")
    
    async def _rate_limited_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Make rate-limited request to FDA API"""
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        # Make request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.session.get(url, params=params, timeout=15)
        )
        
        self.last_request_time = asyncio.get_event_loop().time()
        return response
    
    async def get_drug_recalls(
        self, 
        drug_name: str, 
        limit: int = 20,
        date_range_days: int = 365
    ) -> Dict[str, Any]:
        """Get drug recall information from FDA enforcement database"""
        try:
            logger.info(f"Getting drug recalls for: {drug_name}")
            
            # Use simpler search pattern - just search for the drug name in product description
            # OpenFDA API works better with simple searches
            params = {
                'search': f'product_description:"{drug_name}"',
                'limit': min(limit, 100),  # FDA max is 100
                'sort': 'recall_initiation_date:desc'
            }
            
            logger.debug(f"OpenFDA recall search params: {params}")
            response = await self._rate_limited_request(self.drug_recall_endpoint, params)
            
            if response.status_code == 200:
                data = response.json()
                return await self._process_recall_results(data, drug_name)
            
            elif response.status_code == 404:
                # No results found
                return {
                    'drug_name': drug_name,
                    'success': True,
                    'total_recalls': 0,
                    'recalls': [],
                    'message': 'No recalls found for this drug'
                }
            
            elif response.status_code == 429:
                logger.warning("OpenFDA API rate limit exceeded")
                return {
                    'drug_name': drug_name,
                    'success': False,
                    'error': 'Rate limit exceeded - try again later'
                }
            
            else:
                logger.error(f"OpenFDA recall API error: {response.status_code}")
                logger.debug(f"Response content: {response.text[:500]}")
                return {
                    'drug_name': drug_name,
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except requests.exceptions.Timeout:
            logger.error("OpenFDA recall API request timeout")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': 'Request timeout'
            }
        
        except Exception as e:
            logger.error(f"OpenFDA recall API unexpected error: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': str(e)
            }
    
    async def _process_recall_results(self, data: Dict[str, Any], drug_name: str) -> Dict[str, Any]:
        """Process and format recall results from OpenFDA API"""
        try:
            results = data.get('results', [])
            
            recalls = []
            for recall in results:
                processed_recall = {
                    'recall_number': recall.get('recall_number', ''),
                    'status': recall.get('status', ''),
                    'classification': recall.get('classification', ''),
                    'product_description': recall.get('product_description', ''),
                    'reason_for_recall': recall.get('reason_for_recall', ''),
                    'recall_initiation_date': recall.get('recall_initiation_date', ''),
                    'firm_name': recall.get('recalling_firm', ''),
                    'distribution_pattern': recall.get('distribution_pattern', ''),
                    'product_quantity': recall.get('product_quantity', ''),
                    'voluntary_mandated': recall.get('voluntary_mandated', ''),
                    'initial_firm_notification': recall.get('initial_firm_notification', ''),
                    'event_id': recall.get('event_id', ''),
                    'more_code_info': recall.get('more_code_info', ''),
                    'severity_level': self._determine_severity_level(recall.get('classification', ''))
                }
                recalls.append(processed_recall)
            
            # Generate summary statistics
            summary = self._generate_recall_summary(recalls)
            
            return {
                'drug_name': drug_name,
                'success': True,
                'total_recalls': len(recalls),
                'recalls': recalls,
                'summary': summary,
                'data_source': 'FDA Enforcement Database',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing recall results: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': f'Error processing results: {str(e)}'
            }
    
    def _determine_severity_level(self, classification: str) -> str:
        """Determine severity level from FDA classification"""
        if not classification:
            return 'unknown'
        
        classification_lower = classification.lower()
        
        if 'class i' in classification_lower:
            return 'high'  # Most serious - reasonable probability of serious adverse health consequences or death
        elif 'class ii' in classification_lower:
            return 'medium'  # Temporary or medically reversible adverse health consequences
        elif 'class iii' in classification_lower:
            return 'low'  # Unlikely to cause adverse health consequences
        else:
            return 'unknown'
    
    def _generate_recall_summary(self, recalls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for recalls"""
        if not recalls:
            return {}
        
        # Count by classification
        classifications = {}
        severity_levels = {}
        recent_recalls = 0
        
        current_year = datetime.now().year
        
        for recall in recalls:
            # Classification counts
            classification = recall.get('classification', 'Unknown')
            classifications[classification] = classifications.get(classification, 0) + 1
            
            # Severity level counts
            severity = recall.get('severity_level', 'unknown')
            severity_levels[severity] = severity_levels.get(severity, 0) + 1
            
            # Recent recalls (this year)
            recall_date = recall.get('recall_initiation_date', '')
            if recall_date and recall_date.startswith(str(current_year)):
                recent_recalls += 1
        
        return {
            'total_recalls': len(recalls),
            'recent_recalls_this_year': recent_recalls,
            'classifications': classifications,
            'severity_levels': severity_levels,
            'most_recent_recall': recalls[0].get('recall_initiation_date', '') if recalls else '',
            'has_high_severity': severity_levels.get('high', 0) > 0
        }
    
    async def get_adverse_events(
        self, 
        drug_name: str, 
        limit: int = 20,
        date_range_days: int = 365
    ) -> Dict[str, Any]:
        """Get adverse event information from FDA FAERS database"""
        try:
            logger.info(f"Getting adverse events for: {drug_name}")
            
            # Use simpler search pattern - just search for the drug name in medicinal product
            # OpenFDA API works better with simple searches
            # Broaden search to include multiple fields for better coverage
            # Using a more flexible search query with OR conditions
            search_query = f'patient.drug.medicinalproduct:"{drug_name}" OR patient.drug.openfda.substance_name:"{drug_name}" OR patient.drug.drugcharacterization:"{drug_name}"'
            
            params = {
                'search': search_query,
                'limit': min(limit, 1000),  # Increase limit to retrieve more results, FDA max is 1000
                'sort': 'receiptdate:desc'
            }
            
            logger.debug(f"OpenFDA adverse events search params: {params}")
            response = await self._rate_limited_request(self.drug_event_endpoint, params)
            
            if response.status_code == 200:
                data = response.json()
                return await self._process_adverse_event_results(data, drug_name)
            
            elif response.status_code == 404:
                return {
                    'drug_name': drug_name,
                    'success': True,
                    'total_events': 0,
                    'adverse_events': [],
                    'message': 'No adverse events found for this drug'
                }
            
            elif response.status_code == 429:
                logger.warning("OpenFDA adverse events API rate limit exceeded")
                return {
                    'drug_name': drug_name,
                    'success': False,
                    'error': 'Rate limit exceeded - try again later'
                }
            
            else:
                logger.error(f"OpenFDA adverse events API error: {response.status_code}")
                logger.debug(f"Response content: {response.text[:500]}")
                return {
                    'drug_name': drug_name,
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"OpenFDA adverse events API error: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': str(e)
            }
    
    async def _process_adverse_event_results(self, data: Dict[str, Any], drug_name: str) -> Dict[str, Any]:
        """Process and format adverse event results"""
        try:
            results = data.get('results', [])
            
            adverse_events = []
            all_reactions = []
            
            # Dictionary to store reaction frequencies
            reaction_frequencies = {}
            
            for event in results:
                    patient = event.get('patient', {})
                    
                    # Extract reactions
                    reactions = []
                    if 'reaction' in patient:
                        for reaction in patient['reaction']:
                            reaction_term = reaction.get('reactionmeddrapt', '')
                            if reaction_term:
                                reactions.append(reaction_term)
                                all_reactions.append(reaction_term.lower())
                                # Update reaction frequencies
                                reaction_frequencies[reaction_term] = reaction_frequencies.get(reaction_term, 0) + 1
                
                # Extract patient demographics
                    patient_age = patient.get('patientonsetage', '')
                    patient_sex = patient.get('patientsex', '')
                
                # Map sex codes
                    sex_mapping = {'1': 'Male', '2': 'Female', '0': 'Unknown'}
                    patient_sex_text = sex_mapping.get(patient_sex, 'Unknown')
                
                    # Process event
                    processed_event = {
                    'reactions': reactions,
                    'serious': event.get('serious', ''),
                    'patient_age': patient_age,
                    'patient_sex': patient_sex_text,
                    'report_date': event.get('receiptdate', ''),
                    'outcome': patient.get('patientdeath', {}).get('patientdeathdate', '') if patient.get('patientdeath') else '',
                    'reporter_qualification': event.get('primarysourcecountry', ''),
                    'event_id': event.get('safetyreportid', ''),
                    'severity_assessment': self._assess_event_severity(event)
                    }
                
                    adverse_events.append(processed_event)
            
                    # Generate summary
                    summary = self._generate_adverse_event_summary(adverse_events, all_reactions)
                    
                    # Sort reactions by frequency and get top N
                    sorted_reactions = sorted(reaction_frequencies.items(), key=lambda item: item[1], reverse=True)
                    top_reactions = sorted_reactions[:10]  # Get top 10 most frequent reactions
                    
                    # Format top reactions for display
                    formatted_top_reactions = [
                        f"{reaction.replace('_', ' ').title()} ({count} reports)"
                        for reaction, count in top_reactions
                    ]
                    
                    return {
                        'drug_name': drug_name,
                        'success': True,
                        'total_events': len(adverse_events),
                        'adverse_events': adverse_events,
                        'summary': summary,
                        'data_source': 'FDA Adverse Event Reporting System (FAERS)',
                        'last_updated': datetime.now().isoformat(),
                        'top_reactions': formatted_top_reactions if formatted_top_reactions else ['No specific side effects data available from FDA. Please consult your healthcare provider or pharmacist for complete information.']
                    }
            
        except Exception as e:
            logger.error(f"Error processing adverse event results: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': f'Error processing results: {str(e)}'
            }
    
    def _assess_event_severity(self, event: Dict[str, Any]) -> str:
        """Assess severity of adverse event"""
        serious = event.get('serious', '')
        patient = event.get('patient', {})
        
        # Check for death
        if patient.get('patientdeath'):
            return 'fatal'
        
        # Check for serious designation
        if serious == '1':
            return 'serious'
        
        # Check for hospitalization or life-threatening
        if event.get('seriousnesshospitalization') == '1':
            return 'serious'
        
        if event.get('seriousnesslifethreatening') == '1':
            return 'serious'
        
        return 'non-serious'
    
    def _generate_adverse_event_summary(
        self, 
        events: List[Dict[str, Any]], 
        all_reactions: List[str]
    ) -> Dict[str, Any]:
        """Generate summary statistics for adverse events"""
        if not events:
            return {}
        
        # Count reactions
        reaction_counts = {}
        for reaction in all_reactions:
            reaction_counts[reaction] = reaction_counts.get(reaction, 0) + 1
        
        # Get top 10 most common reactions
        top_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Count serious events
        serious_events = sum(1 for event in events if event.get('serious') == '1')
        fatal_events = sum(1 for event in events if event.get('severity_assessment') == 'fatal')
        
        # Age and sex distribution
        age_groups = {'<18': 0, '18-65': 0, '>65': 0, 'unknown': 0}
        sex_distribution = {'Male': 0, 'Female': 0, 'Unknown': 0}
        
        for event in events:
            # Age grouping
            age = event.get('patient_age', '')
            if age and age.isdigit():
                age_num = int(age)
                if age_num < 18:
                    age_groups['<18'] += 1
                elif age_num <= 65:
                    age_groups['18-65'] += 1
                else:
                    age_groups['>65'] += 1
            else:
                age_groups['unknown'] += 1
            
            # Sex distribution
            sex = event.get('patient_sex', 'Unknown')
            sex_distribution[sex] = sex_distribution.get(sex, 0) + 1
        
        return {
            'total_events': len(events),
            'serious_events': serious_events,
            'fatal_events': fatal_events,
            'top_reactions': top_reactions,
            'age_distribution': age_groups,
            'sex_distribution': sex_distribution,
            'serious_event_rate': round(serious_events / len(events) * 100, 1) if events else 0
        }
    
    async def get_drug_label_info(self, drug_name: str) -> Dict[str, Any]:
        """Get drug labeling information from FDA"""
        try:
            logger.info(f"Getting drug label info for: {drug_name}")
            
            # Use simpler search pattern - just search for the drug name in brand name
            # OpenFDA API works better with simple searches
            params = {
                'search': f'openfda.brand_name:"{drug_name}"',
                'limit': 5
            }
            
            logger.debug(f"OpenFDA label search params: {params}")
            response = await self._rate_limited_request(self.drug_label_endpoint, params)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_label_results(data, drug_name)
            
            elif response.status_code == 404:
                return {
                    'drug_name': drug_name,
                    'success': True,
                    'labels_found': 0,
                    'message': 'No label information found'
                }
            
            else:
                logger.error(f"OpenFDA label API error: {response.status_code}")
                logger.debug(f"Response content: {response.text[:500]}")
                return {
                    'drug_name': drug_name,
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"Error getting drug label info: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': str(e)
            }
    
    def _process_label_results(self, data: Dict[str, Any], drug_name: str) -> Dict[str, Any]:
        """Process drug label results"""
        results = data.get('results', [])
        
        if not results:
            return {
                'drug_name': drug_name,
                'success': True,
                'labels_found': 0
            }
        
        # Extract key information from first result
        label = results[0]
        openfda = label.get('openfda', {})
        
        return {
            'drug_name': drug_name,
            'success': True,
            'labels_found': len(results),
            'brand_names': openfda.get('brand_name', []),
            'generic_names': openfda.get('generic_name', []),
            'manufacturer': openfda.get('manufacturer_name', []),
            'warnings': label.get('warnings', []),
            'indications_and_usage': label.get('indications_and_usage', []),
            'dosage_and_administration': label.get('dosage_and_administration', []),
            'contraindications': label.get('contraindications', [])
        }
    
    async def search_drug_comprehensive(self, drug_name: str) -> Dict[str, Any]:
        """Get comprehensive drug information including recalls, adverse events, and labels"""
        try:
            logger.info(f"Getting comprehensive drug information for: {drug_name}")
            
            # Run all searches concurrently
            recalls_task = self.get_drug_recalls(drug_name)
            events_task = self.get_adverse_events(drug_name)
            labels_task = self.get_drug_label_info(drug_name)
            
            recalls_result, events_result, labels_result = await asyncio.gather(
                recalls_task, events_task, labels_task, return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(recalls_result, Exception):
                recalls_result = {'success': False, 'error': str(recalls_result)}
            
            if isinstance(events_result, Exception):
                events_result = {'success': False, 'error': str(events_result)}
            
            if isinstance(labels_result, Exception):
                labels_result = {'success': False, 'error': str(labels_result)}
            
            # Combine results
            comprehensive_result = {
                'drug_name': drug_name,
                'search_timestamp': datetime.now().isoformat(),
                'recalls': recalls_result,
                'adverse_events': events_result,
                'label_info': labels_result,
                'overall_safety_assessment': self._assess_overall_safety(
                    recalls_result, events_result
                )
            }
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive drug search: {e}")
            return {
                'drug_name': drug_name,
                'success': False,
                'error': str(e)
            }
    
    def _assess_overall_safety(
        self, 
        recalls_result: Dict[str, Any], 
        events_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall safety profile based on recalls and adverse events"""
        assessment = {
            'safety_level': 'unknown',
            'concerns': [],
            'recommendations': []
        }
        
        try:
            # Check recalls
            if recalls_result.get('success') and recalls_result.get('total_recalls', 0) > 0:
                recall_summary = recalls_result.get('summary', {})
                
                if recall_summary.get('has_high_severity'):
                    assessment['safety_level'] = 'high_concern'
                    assessment['concerns'].append('Recent high-severity recalls reported')
                elif recalls_result.get('total_recalls', 0) > 5:
                    assessment['safety_level'] = 'moderate_concern'
                    assessment['concerns'].append('Multiple recent recalls reported')
            
            # Check adverse events
            if events_result.get('success') and events_result.get('total_events', 0) > 0:
                event_summary = events_result.get('summary', {})
                
                if event_summary.get('fatal_events', 0) > 0:
                    assessment['safety_level'] = 'high_concern'
                    assessment['concerns'].append('Fatal adverse events reported')
                elif event_summary.get('serious_event_rate', 0) > 20:
                    if assessment['safety_level'] != 'high_concern':
                        assessment['safety_level'] = 'moderate_concern'
                    assessment['concerns'].append('High rate of serious adverse events')
            
            # Set safety level if no concerns found
            if not assessment['concerns']:
                assessment['safety_level'] = 'no_major_concerns'
            
            # Add recommendations
            if assessment['safety_level'] == 'high_concern':
                assessment['recommendations'].append('Consult healthcare provider immediately about safety concerns')
                assessment['recommendations'].append('Discuss alternative medications if available')
            elif assessment['safety_level'] == 'moderate_concern':
                assessment['recommendations'].append('Discuss safety profile with healthcare provider')
                assessment['recommendations'].append('Monitor for adverse effects closely')
            else:
                assessment['recommendations'].append('Follow standard medication monitoring guidelines')
            
        except Exception as e:
            logger.error(f"Error assessing overall safety: {e}")
            assessment['error'] = str(e)
        
        return assessment
    
    def test_connection(self) -> bool:
        """Test basic connection to OpenFDA API"""
        try:
            # Test with a simple query to the drug label endpoint
            test_params = {
                'search': 'openfda.brand_name:"aspirin"',
                'limit': 1
            }
            
            if self.api_key:
                test_params['api_key'] = self.api_key
            
            response = self.session.get(self.drug_label_endpoint, params=test_params, timeout=10)
            
            if response.status_code == 200:
                logger.info("OpenFDA API connection test: PASSED")
                return True
            else:
                logger.error(f"OpenFDA API connection test failed: {response.status_code}")
                logger.debug(f"Test response: {response.text[:500]}")
                return False
                
        except Exception as e:
            logger.error(f"OpenFDA API connection test error: {e}")
            return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client configuration information"""
        return {
            'base_url': self.base_url,
            'api_key_configured': bool(self.api_key),
            'rate_limit': self.rate_limit,
            'endpoints': {
                'recalls': self.drug_recall_endpoint,
                'adverse_events': self.drug_event_endpoint,
                'labels': self.drug_label_endpoint
            }
        }
    
    def close(self):
        """Close the HTTP session"""
        if self.session:
            self.session.close()
        logger.info("OpenFDA API client closed")