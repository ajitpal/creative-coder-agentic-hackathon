"""
Unit tests for OpenFDA API client
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
from datetime import datetime
from src.openfda_client import OpenFDAClient

class TestOpenFDAClient:
    """Test cases for OpenFDAClient class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_fda_api_key"
    
    @pytest.fixture
    def client(self, mock_api_key):
        """Create OpenFDAClient instance for testing"""
        return OpenFDAClient(api_key=mock_api_key)
    
    @pytest.fixture
    def mock_recall_response(self):
        """Mock recall response from OpenFDA API"""
        return {
            "results": [
                {
                    "recall_number": "F-2024-001",
                    "status": "Ongoing",
                    "classification": "Class I",
                    "product_description": "Test Drug 100mg tablets",
                    "reason_for_recall": "Potential contamination with foreign substance",
                    "recall_initiation_date": "20240115",
                    "recalling_firm": "Test Pharmaceutical Company",
                    "distribution_pattern": "Nationwide",
                    "product_quantity": "10,000 bottles",
                    "voluntary_mandated": "Voluntary",
                    "event_id": "12345"
                },
                {
                    "recall_number": "F-2024-002",
                    "status": "Completed",
                    "classification": "Class II",
                    "product_description": "Test Drug 50mg tablets",
                    "reason_for_recall": "Labeling error",
                    "recall_initiation_date": "20240110",
                    "recalling_firm": "Test Pharmaceutical Company"
                }
            ]
        }
    
    @pytest.fixture
    def mock_adverse_event_response(self):
        """Mock adverse event response from OpenFDA API"""
        return {
            "results": [
                {
                    "safetyreportid": "12345",
                    "serious": "1",
                    "receiptdate": "20240115",
                    "patient": {
                        "patientonsetage": "45",
                        "patientsex": "2",
                        "reaction": [
                            {"reactionmeddrapt": "Nausea"},
                            {"reactionmeddrapt": "Headache"}
                        ]
                    }
                },
                {
                    "safetyreportid": "12346",
                    "serious": "0",
                    "receiptdate": "20240112",
                    "patient": {
                        "patientonsetage": "32",
                        "patientsex": "1",
                        "reaction": [
                            {"reactionmeddrapt": "Dizziness"}
                        ]
                    }
                }
            ]
        }
    
    def test_client_initialization(self, mock_api_key):
        """Test client initialization"""
        client = OpenFDAClient(api_key=mock_api_key)
        
        assert client.api_key == mock_api_key
        assert client.base_url is not None
        assert client.rate_limit == 240  # With API key
        assert client.drug_recall_endpoint.endswith('/drug/enforcement.json')
        assert client.drug_event_endpoint.endswith('/drug/event.json')
    
    def test_client_initialization_without_api_key(self):
        """Test client initialization without API key"""
        with patch('src.config.Config.OPENFDA_API_KEY', None):
            client = OpenFDAClient()
            assert client.api_key is None
            assert client.rate_limit == 40  # Without API key
    
    def test_determine_severity_level(self, client):
        """Test severity level determination from classification"""
        assert client._determine_severity_level("Class I") == "high"
        assert client._determine_severity_level("Class II") == "medium"
        assert client._determine_severity_level("Class III") == "low"
        assert client._determine_severity_level("Unknown") == "unknown"
        assert client._determine_severity_level("") == "unknown"
    
    def test_generate_recall_summary(self, client):
        """Test recall summary generation"""
        recalls = [
            {
                "classification": "Class I",
                "severity_level": "high",
                "recall_initiation_date": "20240115"
            },
            {
                "classification": "Class II",
                "severity_level": "medium",
                "recall_initiation_date": "20240110"
            }
        ]
        
        summary = client._generate_recall_summary(recalls)
        
        assert summary['total_recalls'] == 2
        assert summary['classifications']['Class I'] == 1
        assert summary['classifications']['Class II'] == 1
        assert summary['severity_levels']['high'] == 1
        assert summary['severity_levels']['medium'] == 1
        assert summary['has_high_severity'] is True
    
    def test_assess_event_severity(self, client):
        """Test adverse event severity assessment"""
        # Fatal event
        fatal_event = {
            "patient": {"patientdeath": {"patientdeathdate": "20240115"}}
        }
        assert client._assess_event_severity(fatal_event) == "fatal"
        
        # Serious event
        serious_event = {"serious": "1"}
        assert client._assess_event_severity(serious_event) == "serious"
        
        # Hospitalization
        hospital_event = {"seriousnesshospitalization": "1"}
        assert client._assess_event_severity(hospital_event) == "serious"
        
        # Non-serious event
        non_serious_event = {"serious": "0"}
        assert client._assess_event_severity(non_serious_event) == "non-serious"
    
    def test_generate_adverse_event_summary(self, client):
        """Test adverse event summary generation"""
        events = [
            {
                "serious": "1",
                "severity_assessment": "serious",
                "patient_age": "45",
                "patient_sex": "Female"
            },
            {
                "serious": "0",
                "severity_assessment": "non-serious",
                "patient_age": "32",
                "patient_sex": "Male"
            }
        ]
        
        all_reactions = ["nausea", "headache", "dizziness"]
        
        summary = client._generate_adverse_event_summary(events, all_reactions)
        
        assert summary['total_events'] == 2
        assert summary['serious_events'] == 1
        assert summary['serious_event_rate'] == 50.0
        assert summary['sex_distribution']['Female'] == 1
        assert summary['sex_distribution']['Male'] == 1
    
    @pytest.mark.asyncio
    async def test_get_drug_recalls_success(self, client, mock_recall_response):
        """Test successful drug recall retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_recall_response
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_drug_recalls("test_drug")
        
        assert result['success'] is True
        assert result['drug_name'] == "test_drug"
        assert result['total_recalls'] == 2
        assert len(result['recalls']) == 2
        
        # Check first recall
        first_recall = result['recalls'][0]
        assert first_recall['recall_number'] == "F-2024-001"
        assert first_recall['classification'] == "Class I"
        assert first_recall['severity_level'] == "high"
    
    @pytest.mark.asyncio
    async def test_get_drug_recalls_no_results(self, client):
        """Test drug recall retrieval with no results"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_drug_recalls("nonexistent_drug")
        
        assert result['success'] is True
        assert result['total_recalls'] == 0
        assert result['recalls'] == []
        assert 'No recalls found' in result['message']
    
    @pytest.mark.asyncio
    async def test_get_drug_recalls_rate_limit(self, client):
        """Test drug recall retrieval with rate limit error"""
        mock_response = Mock()
        mock_response.status_code = 429
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_drug_recalls("test_drug")
        
        assert result['success'] is False
        assert 'Rate limit exceeded' in result['error']
    
    @pytest.mark.asyncio
    async def test_get_adverse_events_success(self, client, mock_adverse_event_response):
        """Test successful adverse event retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_adverse_event_response
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_adverse_events("test_drug")
        
        assert result['success'] is True
        assert result['drug_name'] == "test_drug"
        assert result['total_events'] == 2
        assert len(result['adverse_events']) == 2
        
        # Check first event
        first_event = result['adverse_events'][0]
        assert first_event['serious'] == "1"
        assert first_event['patient_sex'] == "Female"  # Code 2 mapped to Female
        assert "Nausea" in first_event['reactions']
        assert "Headache" in first_event['reactions']
    
    @pytest.mark.asyncio
    async def test_get_adverse_events_no_results(self, client):
        """Test adverse event retrieval with no results"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_adverse_events("nonexistent_drug")
        
        assert result['success'] is True
        assert result['total_events'] == 0
        assert result['adverse_events'] == []
    
    @pytest.mark.asyncio
    async def test_get_drug_label_info_success(self, client):
        """Test successful drug label information retrieval"""
        mock_label_response = {
            "results": [
                {
                    "openfda": {
                        "brand_name": ["Test Brand"],
                        "generic_name": ["test_generic"],
                        "manufacturer_name": ["Test Manufacturer"]
                    },
                    "warnings": ["Warning: May cause drowsiness"],
                    "indications_and_usage": ["For treatment of test condition"],
                    "contraindications": ["Do not use if allergic"]
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_label_response
        
        with patch.object(client, '_rate_limited_request', return_value=mock_response):
            result = await client.get_drug_label_info("test_drug")
        
        assert result['success'] is True
        assert result['labels_found'] == 1
        assert "Test Brand" in result['brand_names']
        assert "test_generic" in result['generic_names']
        assert "Warning: May cause drowsiness" in result['warnings']
    
    @pytest.mark.asyncio
    async def test_search_drug_comprehensive(self, client, mock_recall_response, mock_adverse_event_response):
        """Test comprehensive drug search"""
        # Mock all three API calls
        mock_response_recalls = Mock()
        mock_response_recalls.status_code = 200
        mock_response_recalls.json.return_value = mock_recall_response
        
        mock_response_events = Mock()
        mock_response_events.status_code = 200
        mock_response_events.json.return_value = mock_adverse_event_response
        
        mock_response_labels = Mock()
        mock_response_labels.status_code = 404  # No labels found
        
        responses = [mock_response_recalls, mock_response_events, mock_response_labels]
        
        with patch.object(client, '_rate_limited_request', side_effect=responses):
            result = await client.search_drug_comprehensive("test_drug")
        
        assert result['drug_name'] == "test_drug"
        assert 'recalls' in result
        assert 'adverse_events' in result
        assert 'label_info' in result
        assert 'overall_safety_assessment' in result
        
        # Check that recalls and adverse events were processed
        assert result['recalls']['success'] is True
        assert result['adverse_events']['success'] is True
    
    def test_assess_overall_safety_high_concern(self, client):
        """Test overall safety assessment with high concerns"""
        recalls_result = {
            'success': True,
            'total_recalls': 3,
            'summary': {'has_high_severity': True}
        }
        
        events_result = {
            'success': True,
            'total_events': 10,
            'summary': {'fatal_events': 2, 'serious_event_rate': 30}
        }
        
        assessment = client._assess_overall_safety(recalls_result, events_result)
        
        assert assessment['safety_level'] == 'high_concern'
        assert len(assessment['concerns']) > 0
        assert 'Consult healthcare provider immediately' in assessment['recommendations'][0]
    
    def test_assess_overall_safety_moderate_concern(self, client):
        """Test overall safety assessment with moderate concerns"""
        recalls_result = {
            'success': True,
            'total_recalls': 8,
            'summary': {'has_high_severity': False}
        }
        
        events_result = {
            'success': True,
            'total_events': 5,
            'summary': {'fatal_events': 0, 'serious_event_rate': 25}
        }
        
        assessment = client._assess_overall_safety(recalls_result, events_result)
        
        assert assessment['safety_level'] == 'moderate_concern'
        assert 'Multiple recent recalls' in assessment['concerns'][0]
    
    def test_assess_overall_safety_no_concerns(self, client):
        """Test overall safety assessment with no major concerns"""
        recalls_result = {
            'success': True,
            'total_recalls': 0
        }
        
        events_result = {
            'success': True,
            'total_events': 2,
            'summary': {'fatal_events': 0, 'serious_event_rate': 10}
        }
        
        assessment = client._assess_overall_safety(recalls_result, events_result)
        
        assert assessment['safety_level'] == 'no_major_concerns'
        assert len(assessment['concerns']) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limited_request(self, client):
        """Test rate limiting functionality"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(client.session, 'get', return_value=mock_response):
            # First request should go through immediately
            start_time = asyncio.get_event_loop().time()
            await client._rate_limited_request("http://test.com", {"param": "value"})
            first_request_time = asyncio.get_event_loop().time() - start_time
            
            # Second request should be delayed due to rate limiting
            start_time = asyncio.get_event_loop().time()
            await client._rate_limited_request("http://test.com", {"param": "value"})
            second_request_time = asyncio.get_event_loop().time() - start_time
            
            # Second request should take longer due to rate limiting
            assert second_request_time >= client.request_interval
    
    def test_test_connection_success(self, client):
        """Test successful connection test"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.test_connection()
        
        assert result is True
    
    def test_test_connection_no_results(self, client):
        """Test connection test with no results (404 is acceptable)"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.test_connection()
        
        assert result is True  # 404 is acceptable for connection test
    
    def test_test_connection_failure(self, client):
        """Test failed connection test"""
        with patch.object(client.session, 'get', side_effect=Exception("Connection failed")):
            result = client.test_connection()
        
        assert result is False
    
    def test_get_client_info(self, client):
        """Test getting client information"""
        info = client.get_client_info()
        
        assert 'base_url' in info
        assert 'api_key_configured' in info
        assert 'rate_limit' in info
        assert 'endpoints' in info
        assert info['api_key_configured'] is True
        assert info['rate_limit'] == 240
        assert 'recalls' in info['endpoints']
        assert 'adverse_events' in info['endpoints']
        assert 'labels' in info['endpoints']
    
    def test_close_client(self, client):
        """Test closing the client"""
        mock_session = Mock()
        client.session = mock_session
        
        client.close()
        
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_recall_results_empty(self, client):
        """Test processing empty recall results"""
        empty_data = {"results": []}
        
        result = await client._process_recall_results(empty_data, "test_drug")
        
        assert result['success'] is True
        assert result['total_recalls'] == 0
        assert result['recalls'] == []
    
    @pytest.mark.asyncio
    async def test_process_adverse_event_results_empty(self, client):
        """Test processing empty adverse event results"""
        empty_data = {"results": []}
        
        result = await client._process_adverse_event_results(empty_data, "test_drug")
        
        assert result['success'] is True
        assert result['total_events'] == 0
        assert result['adverse_events'] == []
    
    def test_process_label_results_empty(self, client):
        """Test processing empty label results"""
        empty_data = {"results": []}
        
        result = client._process_label_results(empty_data, "test_drug")
        
        assert result['success'] is True
        assert result['labels_found'] == 0