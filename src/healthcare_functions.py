"""
Healthcare-specific functions for Gemini function calling
These functions will be registered with the Gemini client for agentic behavior
"""

from typing import Dict, Any, List, Optional
import asyncio
from src.utils import setup_logging

logger = setup_logging()

# Function parameter schemas for Gemini
FUNCTION_SCHEMAS = {
    "search_medical_term": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "Medical term or disease name to search for"
            }
        },
        "required": ["term"]
    },
    
    "get_drug_information": {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Name of the drug to get information about"
            },
            "include_recalls": {
                "type": "boolean",
                "description": "Whether to include recall information",
                "default": True
            },
            "include_adverse_events": {
                "type": "boolean",
                "description": "Whether to include adverse event information",
                "default": True
            }
        },
        "required": ["drug_name"]
    },
    
    "analyze_symptoms": {
        "type": "object",
        "properties": {
            "symptoms": {
                "type": "string",
                "description": "Description of symptoms to analyze"
            },
            "patient_age": {
                "type": "integer",
                "description": "Patient age (optional)",
                "minimum": 0,
                "maximum": 150
            },
            "patient_gender": {
                "type": "string",
                "description": "Patient gender (optional)",
                "enum": ["male", "female", "other", "unknown"]
            }
        },
        "required": ["symptoms"]
    },
    
    "extract_medical_entities": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract medical entities from"
            },
            "entity_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["disease", "drug", "symptom", "procedure", "condition"]
                },
                "description": "Types of entities to extract (optional, defaults to all)"
            }
        },
        "required": ["text"]
    },
    
    "summarize_medical_document": {
        "type": "object",
        "properties": {
            "document_text": {
                "type": "string",
                "description": "Medical document text to summarize"
            },
            "document_type": {
                "type": "string",
                "description": "Type of document",
                "enum": ["prescription", "lab_report", "medical_note", "research_article", "unknown"]
            },
            "summary_length": {
                "type": "string",
                "description": "Desired summary length",
                "enum": ["brief", "detailed", "comprehensive"],
                "default": "detailed"
            }
        },
        "required": ["document_text"]
    }
}

class HealthcareFunctions:
    """Container for healthcare-specific functions that Gemini can call"""
    
    def __init__(self, api_clients: Dict[str, Any] = None):
        """Initialize with API clients for external services"""
        self.api_clients = api_clients or {}
        logger.info("Healthcare functions initialized")
    
    def set_api_client(self, name: str, client: Any):
        """Set an API client for external service calls"""
        self.api_clients[name] = client
        logger.info(f"Set API client: {name}")
    
    async def search_medical_term(self, term: str) -> Dict[str, Any]:
        """Search for medical term definition using WHO ICD API"""
        try:
            logger.info(f"Searching medical term: {term}")
            
            # Check if WHO ICD client is available
            if 'who_icd' in self.api_clients:
                who_client = self.api_clients['who_icd']
                result = await who_client.search_term(term)
                
                return {
                    "term": term,
                    "source": "WHO_ICD",
                    "definition": result.get('definition', ''),
                    "code": result.get('code', ''),
                    "synonyms": result.get('synonyms', []),
                    "success": True
                }
            else:
                # Fallback response when WHO client not available
                return {
                    "term": term,
                    "source": "fallback",
                    "definition": f"Medical term lookup for '{term}' - WHO ICD API not available",
                    "success": False,
                    "error": "WHO ICD API client not configured"
                }
                
        except Exception as e:
            logger.error(f"Error searching medical term {term}: {e}")
            return {
                "term": term,
                "source": "error",
                "success": False,
                "error": str(e)
            }
    
    async def get_drug_information(
        self, 
        drug_name: str, 
        include_recalls: bool = True, 
        include_adverse_events: bool = True
    ) -> Dict[str, Any]:
        """Get drug information including recalls and adverse events from FDA"""
        try:
            logger.info(f"Getting drug information: {drug_name}")
            
            result = {
                "drug_name": drug_name,
                "source": "OpenFDA",
                "success": True,
                "recalls": [],
                "adverse_events": [],
                "summary": {}
            }
            
            # Check if OpenFDA client is available
            if 'openfda' in self.api_clients:
                fda_client = self.api_clients['openfda']
                
                if include_recalls:
                    recalls = await fda_client.get_drug_recalls(drug_name)
                    result["recalls"] = recalls.get('recalls', [])
                
                if include_adverse_events:
                    adverse_events_data = await fda_client.get_adverse_events(drug_name)
                    result["adverse_events"] = adverse_events_data.get('adverse_events', [])
                    result["top_reactions"] = adverse_events_data.get('top_reactions', [])
                
                # Generate summary
                result["summary"] = {
                    "total_recalls": len(result["recalls"]),
                    "total_adverse_events": len(result["adverse_events"]),
                    "has_safety_concerns": len(result["recalls"]) > 0 or len(result["adverse_events"]) > 0,
                    "top_reactions": result["top_reactions"]
                }
                
            else:
                result["success"] = False
                result["error"] = "OpenFDA API client not configured"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting drug information for {drug_name}: {e}")
            return {
                "drug_name": drug_name,
                "source": "error",
                "success": False,
                "error": str(e)
            }
    
    async def analyze_symptoms(
        self, 
        symptoms: str, 
        patient_age: Optional[int] = None, 
        patient_gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze symptoms and provide general health insights (no diagnosis)"""
        try:
            logger.info(f"Analyzing symptoms: {symptoms[:50]}...")
            
            # This function provides general insights only - no diagnosis
            result = {
                "symptoms": symptoms,
                "patient_info": {
                    "age": patient_age,
                    "gender": patient_gender
                },
                "analysis": {
                    "urgency_level": "low",  # Default to low
                    "general_insights": [],
                    "recommendations": [],
                    "red_flags": []
                },
                "disclaimers": [
                    "This analysis is for informational purposes only",
                    "Not a medical diagnosis or treatment recommendation",
                    "Consult healthcare professionals for medical concerns"
                ],
                "success": True
            }
            
            # Basic symptom analysis (this would be enhanced with medical NLP)
            symptoms_lower = symptoms.lower()
            
            # Check for emergency symptoms
            emergency_keywords = [
                "chest pain", "difficulty breathing", "severe headache", 
                "loss of consciousness", "severe bleeding", "stroke symptoms",
                "heart attack", "severe abdominal pain"
            ]
            
            for keyword in emergency_keywords:
                if keyword in symptoms_lower:
                    result["analysis"]["urgency_level"] = "emergency"
                    result["analysis"]["red_flags"].append(f"Potential emergency symptom detected: {keyword}")
                    result["analysis"]["recommendations"].append("Seek immediate medical attention")
                    break
            
            # Check for concerning symptoms
            concerning_keywords = [
                "persistent fever", "unexplained weight loss", "severe pain",
                "blood in", "difficulty swallowing", "persistent cough"
            ]
            
            for keyword in concerning_keywords:
                if keyword in symptoms_lower:
                    if result["analysis"]["urgency_level"] == "low":
                        result["analysis"]["urgency_level"] = "medium"
                    result["analysis"]["recommendations"].append("Consider consulting a healthcare provider")
                    break
            
            # General insights based on common symptoms
            if "fever" in symptoms_lower:
                result["analysis"]["general_insights"].append("Fever often indicates infection or inflammation")
            
            if "headache" in symptoms_lower:
                result["analysis"]["general_insights"].append("Headaches can have various causes including stress, dehydration, or underlying conditions")
            
            if "fatigue" in symptoms_lower or "tired" in symptoms_lower:
                result["analysis"]["general_insights"].append("Fatigue can be related to sleep, stress, nutrition, or medical conditions")
            
            # Age-specific considerations
            if patient_age:
                if patient_age > 65:
                    result["analysis"]["recommendations"].append("Older adults should monitor symptoms closely and consult healthcare providers promptly")
                elif patient_age < 18:
                    result["analysis"]["recommendations"].append("Pediatric symptoms should be evaluated by healthcare professionals")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            return {
                "symptoms": symptoms,
                "success": False,
                "error": str(e),
                "disclaimers": ["Analysis unavailable due to technical error"]
            }
    
    async def extract_medical_entities(
        self, 
        text: str, 
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract medical entities from text using NLP libraries"""
        try:
            logger.info(f"Extracting medical entities from text: {text[:50]}...")
            
            result = {
                "text": text[:500] + "..." if len(text) > 500 else text,  # Truncate for response
                "entities": [],
                "entity_counts": {},
                "success": True
            }
            
            # Check if medical NLP client is available
            if 'medical_nlp' in self.api_clients:
                nlp_client = self.api_clients['medical_nlp']
                entities = await nlp_client.extract_entities(text, entity_types)
                result["entities"] = entities
                
                # Count entities by type
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    result["entity_counts"][entity_type] = result["entity_counts"].get(entity_type, 0) + 1
            else:
                # Basic keyword-based entity extraction as fallback
                result["entities"] = self._basic_entity_extraction(text, entity_types)
                
                # Count entities
                for entity in result["entities"]:
                    entity_type = entity.get('type', 'unknown')
                    result["entity_counts"][entity_type] = result["entity_counts"].get(entity_type, 0) + 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "success": False,
                "error": str(e)
            }
    
    def _basic_entity_extraction(self, text: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Basic keyword-based entity extraction as fallback"""
        entities = []
        text_lower = text.lower()
        
        # Basic medical terms (this would be much more comprehensive in production)
        medical_terms = {
            "disease": ["diabetes", "hypertension", "cancer", "pneumonia", "asthma", "arthritis"],
            "drug": ["insulin", "aspirin", "ibuprofen", "acetaminophen", "metformin", "lisinopril"],
            "symptom": ["fever", "headache", "nausea", "fatigue", "cough", "pain", "dizziness"],
            "procedure": ["surgery", "biopsy", "x-ray", "mri", "ct scan", "blood test"]
        }
        
        target_types = entity_types or list(medical_terms.keys())
        
        for entity_type in target_types:
            if entity_type in medical_terms:
                for term in medical_terms[entity_type]:
                    if term in text_lower:
                        start_pos = text_lower.find(term)
                        entities.append({
                            "text": term,
                            "type": entity_type,
                            "confidence": 0.7,  # Basic confidence score
                            "start_pos": start_pos,
                            "end_pos": start_pos + len(term),
                            "source": "keyword_matching"
                        })
        
        return entities
    
    async def summarize_medical_document(
        self, 
        document_text: str, 
        document_type: str = "unknown",
        summary_length: str = "detailed"
    ) -> Dict[str, Any]:
        """Summarize medical document using NLP libraries"""
        try:
            logger.info(f"Summarizing {document_type} document ({len(document_text)} chars)")
            
            result = {
                "document_type": document_type,
                "original_length": len(document_text),
                "summary": "",
                "key_findings": [],
                "extracted_entities": [],
                "confidence_score": 0.0,
                "success": True
            }
            
            # Check if document summarization client is available
            if 'document_summarizer' in self.api_clients:
                summarizer = self.api_clients['document_summarizer']
                summary_result = await summarizer.summarize(document_text, document_type, summary_length)
                
                result.update(summary_result)
            else:
                # Basic summarization fallback
                result["summary"] = self._basic_document_summary(document_text, summary_length)
                result["confidence_score"] = 0.5  # Lower confidence for basic summary
            
            # Extract entities from the document
            entity_result = await self.extract_medical_entities(document_text)
            if entity_result.get("success"):
                result["extracted_entities"] = entity_result.get("entities", [])
            
            return result
            
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return {
                "document_type": document_type,
                "success": False,
                "error": str(e)
            }
    
    def _basic_document_summary(self, text: str, length: str = "detailed") -> str:
        """Basic document summarization as fallback"""
        sentences = text.split('. ')
        
        if length == "brief":
            max_sentences = min(2, len(sentences))
        elif length == "comprehensive":
            max_sentences = min(10, len(sentences))
        else:  # detailed
            max_sentences = min(5, len(sentences))
        
        # Take first few sentences as basic summary
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def get_function_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all function definitions for registration with Gemini"""
        return {
            "search_medical_term": {
                "description": "Search for medical term or disease definition using authoritative medical databases",
                "parameters": FUNCTION_SCHEMAS["search_medical_term"],
                "function": self.search_medical_term
            },
            "get_drug_information": {
                "description": "Get drug information including recalls and adverse events from FDA databases",
                "parameters": FUNCTION_SCHEMAS["get_drug_information"],
                "function": self.get_drug_information
            },
            "analyze_symptoms": {
                "description": "Analyze symptoms and provide general health insights (not medical diagnosis)",
                "parameters": FUNCTION_SCHEMAS["analyze_symptoms"],
                "function": self.analyze_symptoms
            },
            "extract_medical_entities": {
                "description": "Extract medical entities (diseases, drugs, symptoms) from text",
                "parameters": FUNCTION_SCHEMAS["extract_medical_entities"],
                "function": self.extract_medical_entities
            },
            "summarize_medical_document": {
                "description": "Summarize medical documents like prescriptions, reports, or research articles",
                "parameters": FUNCTION_SCHEMAS["summarize_medical_document"],
                "function": self.summarize_medical_document
            }
        }