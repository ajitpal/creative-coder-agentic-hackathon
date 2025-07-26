"""
Data models for Intelligent Healthcare Navigator
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json

class QueryType(Enum):
    """Enumeration of supported query types"""
    MEDICAL_TERM = "medical_term"
    DRUG_INFO = "drug_info"
    SYMPTOMS = "symptoms"
    DOCUMENT_SUMMARY = "document_summary"
    ENTITY_EXTRACTION = "entity_extraction"

class EntityType(Enum):
    """Enumeration of medical entity types"""
    DISEASE = "disease"
    DRUG = "drug"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    CONDITION = "condition"

class DocumentType(Enum):
    """Enumeration of document types"""
    PRESCRIPTION = "prescription"
    LAB_REPORT = "lab_report"
    MEDICAL_NOTE = "medical_note"
    RESEARCH_ARTICLE = "research_article"
    UNKNOWN = "unknown"

@dataclass
class MedicalQuery:
    """Data model for medical queries"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    query_type: QueryType = QueryType.MEDICAL_TERM
    timestamp: datetime = field(default_factory=datetime.now)
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate query data after initialization"""
        if not self.query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if len(self.query_text) > 5000:
            raise ValueError("Query text too long (max 5000 characters)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'query_text': self.query_text,
            'query_type': self.query_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_context': self.user_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalQuery':
        """Create instance from dictionary"""
        return cls(
            id=data['id'],
            query_text=data['query_text'],
            query_type=QueryType(data['query_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_context=data.get('user_context', {})
        )

@dataclass
class MedicalEntity:
    """Data model for extracted medical entities"""
    text: str
    entity_type: EntityType
    confidence: float = 0.0
    source_api: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    
    def __post_init__(self):
        """Validate entity data"""
        if not self.text.strip():
            raise ValueError("Entity text cannot be empty")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'entity_type': self.entity_type.value,
            'confidence': self.confidence,
            'source_api': self.source_api,
            'additional_info': self.additional_info,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos
        }

@dataclass
class MedicalResponse:
    """Data model for medical query responses"""
    query_id: str
    response_text: str
    sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    medical_entities: List[MedicalEntity] = field(default_factory=list)
    disclaimers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate response data"""
        if not self.response_text.strip():
            raise ValueError("Response text cannot be empty")
        
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    def add_disclaimer(self, disclaimer: str):
        """Add medical disclaimer to response"""
        if disclaimer not in self.disclaimers:
            self.disclaimers.append(disclaimer)
    
    def add_entity(self, entity: MedicalEntity):
        """Add medical entity to response"""
        self.medical_entities.append(entity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'query_id': self.query_id,
            'response_text': self.response_text,
            'sources': self.sources,
            'confidence_score': self.confidence_score,
            'medical_entities': [entity.to_dict() for entity in self.medical_entities],
            'disclaimers': self.disclaimers,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DocumentSummary:
    """Data model for medical document summaries"""
    original_text: str
    summary: str
    key_findings: List[str] = field(default_factory=list)
    extracted_entities: List[MedicalEntity] = field(default_factory=list)
    document_type: DocumentType = DocumentType.UNKNOWN
    confidence_score: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate document summary data"""
        if not self.original_text.strip():
            raise ValueError("Original text cannot be empty")
        
        if not self.summary.strip():
            raise ValueError("Summary cannot be empty")
    
    def add_key_finding(self, finding: str):
        """Add key finding to summary"""
        if finding and finding not in self.key_findings:
            self.key_findings.append(finding)
    
    def add_entity(self, entity: MedicalEntity):
        """Add extracted entity to summary"""
        self.extracted_entities.append(entity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'original_text': self.original_text[:500] + "..." if len(self.original_text) > 500 else self.original_text,
            'summary': self.summary,
            'key_findings': self.key_findings,
            'extracted_entities': [entity.to_dict() for entity in self.extracted_entities],
            'document_type': self.document_type.value,
            'confidence_score': self.confidence_score,
            'processing_metadata': self.processing_metadata,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ErrorResponse:
    """Data model for error responses"""
    error_code: str
    error_message: str
    user_message: str
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_code': self.error_code,
            'error_message': self.error_message,
            'user_message': self.user_message,
            'suggested_actions': self.suggested_actions,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }

# API Response Wrapper Classes

@dataclass
class WHOICDResponse:
    """Wrapper for WHO ICD API responses"""
    title: str = ""
    definition: str = ""
    code: str = ""
    synonyms: List[str] = field(default_factory=list)
    parent_categories: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any]) -> 'WHOICDResponse':
        """Create instance from WHO ICD API response"""
        try:
            # Handle nested title structure
            title = ""
            if 'title' in api_response:
                if isinstance(api_response['title'], dict):
                    title = api_response['title'].get('@value', '')
                else:
                    title = str(api_response['title'])
            
            # Handle nested definition structure
            definition = ""
            if 'definition' in api_response:
                if isinstance(api_response['definition'], dict):
                    definition = api_response['definition'].get('@value', '')
                else:
                    definition = str(api_response['definition'])
            
            # Extract code
            code = api_response.get('code', api_response.get('@id', ''))
            
            # Extract synonyms if available
            synonyms = []
            if 'synonym' in api_response:
                synonym_data = api_response['synonym']
                if isinstance(synonym_data, list):
                    synonyms = [s.get('@value', str(s)) if isinstance(s, dict) else str(s) for s in synonym_data]
                elif isinstance(synonym_data, dict):
                    synonyms = [synonym_data.get('@value', str(synonym_data))]
            
            return cls(
                title=title,
                definition=definition,
                code=code,
                synonyms=synonyms,
                raw_response=api_response
            )
        except Exception as e:
            # Return empty response with error info in raw_response
            return cls(raw_response={'error': str(e), 'original_response': api_response})
    
    def is_valid(self) -> bool:
        """Check if response contains valid data"""
        return bool(self.title or self.definition)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'definition': self.definition,
            'code': self.code,
            'synonyms': self.synonyms,
            'parent_categories': self.parent_categories
        }

@dataclass
class OpenFDAResponse:
    """Wrapper for OpenFDA API responses"""
    drug_name: str = ""
    recalls: List[Dict[str, Any]] = field(default_factory=list)
    adverse_events: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    meta_info: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any], query_drug: str = "") -> 'OpenFDAResponse':
        """Create instance from OpenFDA API response"""
        try:
            results = api_response.get('results', [])
            meta = api_response.get('meta', {})
            
            # Extract recall information
            recalls = []
            adverse_events = []
            
            for result in results:
                # Check if this is a recall or adverse event
                if 'recall_number' in result or 'classification' in result:
                    recalls.append({
                        'recall_number': result.get('recall_number', ''),
                        'status': result.get('status', ''),
                        'classification': result.get('classification', ''),
                        'product_description': result.get('product_description', ''),
                        'reason_for_recall': result.get('reason_for_recall', ''),
                        'recall_initiation_date': result.get('recall_initiation_date', ''),
                        'firm_name': result.get('recalling_firm', '')
                    })
                elif 'patient' in result or 'reaction' in result:
                    # This is an adverse event
                    reactions = []
                    if 'patient' in result and 'reaction' in result['patient']:
                        reactions = [r.get('reactionmeddrapt', '') for r in result['patient']['reaction']]
                    
                    adverse_events.append({
                        'reactions': reactions,
                        'serious': result.get('serious', ''),
                        'patient_age': result.get('patient', {}).get('patientonsetage', ''),
                        'patient_sex': result.get('patient', {}).get('patientsex', ''),
                        'report_date': result.get('receiptdate', '')
                    })
            
            total_count = meta.get('results', {}).get('total', len(results))
            
            return cls(
                drug_name=query_drug,
                recalls=recalls,
                adverse_events=adverse_events,
                total_count=total_count,
                meta_info=meta,
                raw_response=api_response
            )
        except Exception as e:
            return cls(
                drug_name=query_drug,
                raw_response={'error': str(e), 'original_response': api_response}
            )
    
    def has_recalls(self) -> bool:
        """Check if response contains recall information"""
        return len(self.recalls) > 0
    
    def has_adverse_events(self) -> bool:
        """Check if response contains adverse event information"""
        return len(self.adverse_events) > 0
    
    def get_recall_summary(self) -> Dict[str, Any]:
        """Get summary of recall information"""
        if not self.has_recalls():
            return {'count': 0, 'classifications': [], 'recent_recalls': []}
        
        classifications = [recall.get('classification', 'Unknown') for recall in self.recalls]
        classification_counts = {}
        for classification in classifications:
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        # Get most recent recalls (first 5)
        recent_recalls = self.recalls[:5]
        
        return {
            'count': len(self.recalls),
            'classifications': classification_counts,
            'recent_recalls': recent_recalls
        }
    
    def get_adverse_event_summary(self) -> Dict[str, Any]:
        """Get summary of adverse event information"""
        if not self.has_adverse_events():
            return {'count': 0, 'common_reactions': [], 'serious_events': 0}
        
        # Count common reactions
        all_reactions = []
        serious_count = 0
        
        for event in self.adverse_events:
            all_reactions.extend(event.get('reactions', []))
            if event.get('serious') == '1':
                serious_count += 1
        
        # Count reaction frequencies
        reaction_counts = {}
        for reaction in all_reactions:
            if reaction:  # Skip empty reactions
                reaction_counts[reaction] = reaction_counts.get(reaction, 0) + 1
        
        # Get top 10 most common reactions
        common_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'count': len(self.adverse_events),
            'common_reactions': common_reactions,
            'serious_events': serious_count
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'drug_name': self.drug_name,
            'recall_summary': self.get_recall_summary(),
            'adverse_event_summary': self.get_adverse_event_summary(),
            'total_count': self.total_count
        }

@dataclass
class GeminiResponse:
    """Wrapper for Google Gemini API responses"""
    text: str = ""
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    usage_metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, api_response: Any) -> 'GeminiResponse':
        """Create instance from Gemini API response"""
        try:
            # Handle different response formats from Gemini API
            text = ""
            function_calls = []
            reasoning = ""
            usage_metadata = {}
            
            # Extract text content
            if hasattr(api_response, 'text'):
                text = api_response.text
            elif hasattr(api_response, 'candidates') and api_response.candidates:
                candidate = api_response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text += part.text
                        elif hasattr(part, 'function_call'):
                            function_calls.append({
                                'name': part.function_call.name,
                                'args': dict(part.function_call.args)
                            })
            
            # Extract usage metadata if available
            if hasattr(api_response, 'usage_metadata'):
                usage_metadata = {
                    'prompt_token_count': getattr(api_response.usage_metadata, 'prompt_token_count', 0),
                    'candidates_token_count': getattr(api_response.usage_metadata, 'candidates_token_count', 0),
                    'total_token_count': getattr(api_response.usage_metadata, 'total_token_count', 0)
                }
            
            return cls(
                text=text,
                function_calls=function_calls,
                reasoning=reasoning,
                usage_metadata=usage_metadata,
                raw_response=str(api_response)  # Convert to string for serialization
            )
        except Exception as e:
            return cls(
                text="",
                raw_response={'error': str(e)}
            )
    
    def has_function_calls(self) -> bool:
        """Check if response contains function calls"""
        return len(self.function_calls) > 0
    
    def get_function_call_names(self) -> List[str]:
        """Get list of function call names"""
        return [call.get('name', '') for call in self.function_calls]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'function_calls': self.function_calls,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'usage_metadata': self.usage_metadata
        }