"""
Unit tests for data models
"""

import pytest
from datetime import datetime
from src.models import (
    MedicalQuery, MedicalResponse, MedicalEntity, DocumentSummary, ErrorResponse,
    QueryType, EntityType, DocumentType
)

class TestMedicalQuery:
    """Test cases for MedicalQuery model"""
    
    def test_valid_query_creation(self):
        """Test creating a valid medical query"""
        query = MedicalQuery(
            query_text="What is diabetes?",
            query_type=QueryType.MEDICAL_TERM
        )
        
        assert query.query_text == "What is diabetes?"
        assert query.query_type == QueryType.MEDICAL_TERM
        assert isinstance(query.id, str)
        assert isinstance(query.timestamp, datetime)
    
    def test_empty_query_text_raises_error(self):
        """Test that empty query text raises ValueError"""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            MedicalQuery(query_text="")
    
    def test_long_query_text_raises_error(self):
        """Test that overly long query text raises ValueError"""
        long_text = "x" * 5001
        with pytest.raises(ValueError, match="Query text too long"):
            MedicalQuery(query_text=long_text)
    
    def test_query_serialization(self):
        """Test query to_dict and from_dict methods"""
        original_query = MedicalQuery(
            query_text="What is hypertension?",
            query_type=QueryType.MEDICAL_TERM,
            user_context={"session_id": "123"}
        )
        
        query_dict = original_query.to_dict()
        restored_query = MedicalQuery.from_dict(query_dict)
        
        assert restored_query.query_text == original_query.query_text
        assert restored_query.query_type == original_query.query_type
        assert restored_query.user_context == original_query.user_context

class TestMedicalEntity:
    """Test cases for MedicalEntity model"""
    
    def test_valid_entity_creation(self):
        """Test creating a valid medical entity"""
        entity = MedicalEntity(
            text="diabetes",
            entity_type=EntityType.DISEASE,
            confidence=0.95,
            source_api="WHO_ICD"
        )
        
        assert entity.text == "diabetes"
        assert entity.entity_type == EntityType.DISEASE
        assert entity.confidence == 0.95
        assert entity.source_api == "WHO_ICD"
    
    def test_empty_entity_text_raises_error(self):
        """Test that empty entity text raises ValueError"""
        with pytest.raises(ValueError, match="Entity text cannot be empty"):
            MedicalEntity(text="", entity_type=EntityType.DISEASE)
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError"""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            MedicalEntity(text="diabetes", entity_type=EntityType.DISEASE, confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            MedicalEntity(text="diabetes", entity_type=EntityType.DISEASE, confidence=-0.1)
    
    def test_entity_serialization(self):
        """Test entity to_dict method"""
        entity = MedicalEntity(
            text="insulin",
            entity_type=EntityType.DRUG,
            confidence=0.88,
            source_api="FDA",
            start_pos=10,
            end_pos=17
        )
        
        entity_dict = entity.to_dict()
        
        assert entity_dict['text'] == "insulin"
        assert entity_dict['entity_type'] == "drug"
        assert entity_dict['confidence'] == 0.88
        assert entity_dict['start_pos'] == 10
        assert entity_dict['end_pos'] == 17

class TestMedicalResponse:
    """Test cases for MedicalResponse model"""
    
    def test_valid_response_creation(self):
        """Test creating a valid medical response"""
        response = MedicalResponse(
            query_id="test-123",
            response_text="Diabetes is a metabolic disorder...",
            sources=["WHO_ICD", "Gemini"],
            confidence_score=0.92
        )
        
        assert response.query_id == "test-123"
        assert "Diabetes is a metabolic disorder" in response.response_text
        assert response.sources == ["WHO_ICD", "Gemini"]
        assert response.confidence_score == 0.92
    
    def test_empty_response_text_raises_error(self):
        """Test that empty response text raises ValueError"""
        with pytest.raises(ValueError, match="Response text cannot be empty"):
            MedicalResponse(query_id="test", response_text="")
    
    def test_add_disclaimer(self):
        """Test adding disclaimers to response"""
        response = MedicalResponse(query_id="test", response_text="Test response")
        
        disclaimer = "This is not medical advice"
        response.add_disclaimer(disclaimer)
        
        assert disclaimer in response.disclaimers
        
        # Test duplicate disclaimer is not added
        response.add_disclaimer(disclaimer)
        assert response.disclaimers.count(disclaimer) == 1
    
    def test_add_entity(self):
        """Test adding medical entities to response"""
        response = MedicalResponse(query_id="test", response_text="Test response")
        entity = MedicalEntity(text="diabetes", entity_type=EntityType.DISEASE)
        
        response.add_entity(entity)
        
        assert len(response.medical_entities) == 1
        assert response.medical_entities[0].text == "diabetes"

class TestDocumentSummary:
    """Test cases for DocumentSummary model"""
    
    def test_valid_summary_creation(self):
        """Test creating a valid document summary"""
        summary = DocumentSummary(
            original_text="Patient presents with elevated blood glucose...",
            summary="Patient has diabetes symptoms",
            document_type=DocumentType.MEDICAL_NOTE
        )
        
        assert "Patient presents with elevated" in summary.original_text
        assert summary.summary == "Patient has diabetes symptoms"
        assert summary.document_type == DocumentType.MEDICAL_NOTE
    
    def test_empty_original_text_raises_error(self):
        """Test that empty original text raises ValueError"""
        with pytest.raises(ValueError, match="Original text cannot be empty"):
            DocumentSummary(original_text="", summary="Test summary")
    
    def test_empty_summary_raises_error(self):
        """Test that empty summary raises ValueError"""
        with pytest.raises(ValueError, match="Summary cannot be empty"):
            DocumentSummary(original_text="Test text", summary="")
    
    def test_add_key_finding(self):
        """Test adding key findings to summary"""
        summary = DocumentSummary(
            original_text="Test text",
            summary="Test summary"
        )
        
        finding = "Elevated glucose levels"
        summary.add_key_finding(finding)
        
        assert finding in summary.key_findings
        
        # Test duplicate finding is not added
        summary.add_key_finding(finding)
        assert summary.key_findings.count(finding) == 1
    
    def test_serialization_truncates_long_text(self):
        """Test that serialization truncates very long original text"""
        long_text = "x" * 600
        summary = DocumentSummary(
            original_text=long_text,
            summary="Test summary"
        )
        
        summary_dict = summary.to_dict()
        
        assert len(summary_dict['original_text']) <= 503  # 500 + "..."
        assert summary_dict['original_text'].endswith("...")

class TestErrorResponse:
    """Test cases for ErrorResponse model"""
    
    def test_valid_error_creation(self):
        """Test creating a valid error response"""
        error = ErrorResponse(
            error_code="API_ERROR",
            error_message="Failed to connect to WHO API",
            user_message="Unable to retrieve medical information at this time",
            suggested_actions=["Try again later", "Check internet connection"]
        )
        
        assert error.error_code == "API_ERROR"
        assert error.error_message == "Failed to connect to WHO API"
        assert len(error.suggested_actions) == 2
        assert isinstance(error.timestamp, datetime)
    
    def test_error_serialization(self):
        """Test error response serialization"""
        error = ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Invalid input",
            user_message="Please check your input"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['error_code'] == "VALIDATION_ERROR"
        assert error_dict['error_message'] == "Invalid input"
        assert 'timestamp' in error_dict
class TestW
HOICDResponse:
    """Test cases for WHOICDResponse model"""
    
    def test_from_api_response_with_nested_structure(self):
        """Test creating WHOICDResponse from nested API response"""
        api_response = {
            'title': {'@value': 'Diabetes mellitus'},
            'definition': {'@value': 'A group of metabolic disorders characterized by high blood sugar'},
            'code': 'E10-E14',
            'synonym': [
                {'@value': 'Diabetes'},
                {'@value': 'DM'}
            ]
        }
        
        response = WHOICDResponse.from_api_response(api_response)
        
        assert response.title == 'Diabetes mellitus'
        assert 'metabolic disorders' in response.definition
        assert response.code == 'E10-E14'
        assert 'Diabetes' in response.synonyms
        assert 'DM' in response.synonyms
        assert response.is_valid()
    
    def test_from_api_response_with_simple_structure(self):
        """Test creating WHOICDResponse from simple API response"""
        api_response = {
            'title': 'Hypertension',
            'definition': 'High blood pressure',
            '@id': 'I10'
        }
        
        response = WHOICDResponse.from_api_response(api_response)
        
        assert response.title == 'Hypertension'
        assert response.definition == 'High blood pressure'
        assert response.code == 'I10'
        assert response.is_valid()
    
    def test_from_api_response_with_error(self):
        """Test handling malformed API response"""
        api_response = {'invalid': 'data'}
        
        response = WHOICDResponse.from_api_response(api_response)
        
        assert not response.is_valid()
        assert 'error' in response.raw_response
    
    def test_serialization(self):
        """Test WHOICDResponse serialization"""
        response = WHOICDResponse(
            title='Test Disease',
            definition='Test definition',
            code='T01',
            synonyms=['Synonym1', 'Synonym2']
        )
        
        response_dict = response.to_dict()
        
        assert response_dict['title'] == 'Test Disease'
        assert response_dict['definition'] == 'Test definition'
        assert response_dict['code'] == 'T01'
        assert len(response_dict['synonyms']) == 2

class TestOpenFDAResponse:
    """Test cases for OpenFDAResponse model"""
    
    def test_from_api_response_with_recalls(self):
        """Test creating OpenFDAResponse from recall data"""
        api_response = {
            'results': [
                {
                    'recall_number': 'F-2024-001',
                    'status': 'Ongoing',
                    'classification': 'Class I',
                    'product_description': 'Test Drug 100mg',
                    'reason_for_recall': 'Contamination',
                    'recall_initiation_date': '20240101',
                    'recalling_firm': 'Test Pharma'
                }
            ],
            'meta': {
                'results': {'total': 1}
            }
        }
        
        response = OpenFDAResponse.from_api_response(api_response, 'Test Drug')
        
        assert response.drug_name == 'Test Drug'
        assert response.has_recalls()
        assert not response.has_adverse_events()
        assert response.total_count == 1
        
        recall_summary = response.get_recall_summary()
        assert recall_summary['count'] == 1
        assert 'Class I' in recall_summary['classifications']
    
    def test_from_api_response_with_adverse_events(self):
        """Test creating OpenFDAResponse from adverse event data"""
        api_response = {
            'results': [
                {
                    'patient': {
                        'reaction': [
                            {'reactionmeddrapt': 'Nausea'},
                            {'reactionmeddrapt': 'Headache'}
                        ],
                        'patientonsetage': '45',
                        'patientsex': '2'
                    },
                    'serious': '1',
                    'receiptdate': '20240101'
                }
            ],
            'meta': {
                'results': {'total': 1}
            }
        }
        
        response = OpenFDAResponse.from_api_response(api_response, 'Test Drug')
        
        assert response.drug_name == 'Test Drug'
        assert not response.has_recalls()
        assert response.has_adverse_events()
        
        ae_summary = response.get_adverse_event_summary()
        assert ae_summary['count'] == 1
        assert ae_summary['serious_events'] == 1
        assert len(ae_summary['common_reactions']) == 2
    
    def test_empty_response(self):
        """Test handling empty API response"""
        api_response = {'results': [], 'meta': {}}
        
        response = OpenFDAResponse.from_api_response(api_response, 'Unknown Drug')
        
        assert response.drug_name == 'Unknown Drug'
        assert not response.has_recalls()
        assert not response.has_adverse_events()
        assert response.total_count == 0
    
    def test_serialization(self):
        """Test OpenFDAResponse serialization"""
        response = OpenFDAResponse(
            drug_name='Test Drug',
            recalls=[{'recall_number': 'F-001', 'classification': 'Class II'}],
            total_count=1
        )
        
        response_dict = response.to_dict()
        
        assert response_dict['drug_name'] == 'Test Drug'
        assert response_dict['recall_summary']['count'] == 1
        assert response_dict['total_count'] == 1

class TestGeminiResponse:
    """Test cases for GeminiResponse model"""
    
    def test_basic_response_creation(self):
        """Test creating basic GeminiResponse"""
        response = GeminiResponse(
            text="This is a test response",
            confidence=0.95,
            usage_metadata={'total_token_count': 100}
        )
        
        assert response.text == "This is a test response"
        assert response.confidence == 0.95
        assert not response.has_function_calls()
    
    def test_response_with_function_calls(self):
        """Test GeminiResponse with function calls"""
        response = GeminiResponse(
            text="I'll search for that information",
            function_calls=[
                {'name': 'search_medical_term', 'args': {'term': 'diabetes'}},
                {'name': 'get_drug_info', 'args': {'drug': 'insulin'}}
            ]
        )
        
        assert response.has_function_calls()
        assert len(response.function_calls) == 2
        
        function_names = response.get_function_call_names()
        assert 'search_medical_term' in function_names
        assert 'get_drug_info' in function_names
    
    def test_serialization(self):
        """Test GeminiResponse serialization"""
        response = GeminiResponse(
            text="Test response",
            function_calls=[{'name': 'test_function', 'args': {}}],
            reasoning="Test reasoning",
            confidence=0.8
        )
        
        response_dict = response.to_dict()
        
        assert response_dict['text'] == "Test response"
        assert len(response_dict['function_calls']) == 1
        assert response_dict['reasoning'] == "Test reasoning"
        assert response_dict['confidence'] == 0.8