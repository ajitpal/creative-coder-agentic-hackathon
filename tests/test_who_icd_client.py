"""
Unit tests for WHO ICD API client
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json
from src.who_icd_client import WHOICDClient
from src.models import WHOICDResponse

class TestWHOICDClient:
    """Test cases for WHOICDClient class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_who_api_key"
    
    @pytest.fixture
    def client(self, mock_api_key):
        """Create WHOICDClient instance for testing"""
        return WHOICDClient(api_key=mock_api_key)
    
    @pytest.fixture
    def mock_search_response(self):
        """Mock successful search response from WHO ICD API"""
        return {
            "destinationEntities": [
                {
                    "@id": "http://id.who.int/icd/entity/123456",
                    "title": {"@value": "Diabetes mellitus"},
                    "definition": {"@value": "A group of metabolic disorders characterized by high blood sugar"},
                    "code": "E10-E14",
                    "synonym": [
                        {"@value": "Diabetes"},
                        {"@value": "DM"}
                    ]
                },
                {
                    "@id": "http://id.who.int/icd/entity/789012",
                    "title": {"@value": "Type 1 diabetes mellitus"},
                    "definition": {"@value": "Diabetes mellitus due to autoimmune destruction of pancreatic beta cells"},
                    "code": "E10"
                }
            ]
        }
    
    def test_client_initialization(self, mock_api_key):
        """Test client initialization"""
        client = WHOICDClient(api_key=mock_api_key)
        
        assert client.api_key == mock_api_key
        assert client.base_url is not None
        assert client.search_endpoint.endswith('/search')
        assert client.entity_endpoint.endswith('/entity')
        assert 'Authorization' in client.session.headers
    
    def test_client_initialization_without_api_key(self):
        """Test client initialization without API key"""
        with patch('src.config.Config.WHO_ICD_API_KEY', None):
            client = WHOICDClient()
            assert client.api_key is None
            assert 'Authorization' not in client.session.headers
    
    @pytest.mark.asyncio
    async def test_search_term_success(self, client, mock_search_response):
        """Test successful term search"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.search_term("diabetes")
        
        assert result['success'] is True
        assert result['term'] == "diabetes"
        assert result['total_results'] == 2
        assert result['returned_results'] == 2
        assert len(result['results']) == 2
        
        # Check best match
        assert result['best_match'] is not None
        assert result['definition'] != ""
        assert result['code'] != ""
    
    @pytest.mark.asyncio
    async def test_search_term_authentication_error(self, client):
        """Test search with authentication error"""
        mock_response = Mock()
        mock_response.status_code = 401
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.search_term("diabetes")
        
        assert result['success'] is False
        assert 'Authentication failed' in result['error']
        assert result['results'] == []
    
    @pytest.mark.asyncio
    async def test_search_term_rate_limit(self, client):
        """Test search with rate limit error"""
        mock_response = Mock()
        mock_response.status_code = 429
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.search_term("diabetes")
        
        assert result['success'] is False
        assert 'Rate limit exceeded' in result['error']
    
    @pytest.mark.asyncio
    async def test_search_term_connection_error(self, client):
        """Test search with connection error"""
        import requests
        
        with patch.object(client.session, 'get', side_effect=requests.exceptions.ConnectionError()):
            result = await client.search_term("diabetes")
        
        assert result['success'] is False
        assert 'Connection error' in result['error']
    
    @pytest.mark.asyncio
    async def test_search_term_timeout(self, client):
        """Test search with timeout error"""
        import requests
        
        with patch.object(client.session, 'get', side_effect=requests.exceptions.Timeout()):
            result = await client.search_term("diabetes")
        
        assert result['success'] is False
        assert 'Request timeout' in result['error']
    
    def test_calculate_match_score(self, client):
        """Test match score calculation"""
        # Exact match
        score = client._calculate_match_score("diabetes", "diabetes")
        assert score == 1.0
        
        # Title starts with search term
        score = client._calculate_match_score("diabetes", "diabetes mellitus")
        assert score == 0.9
        
        # Search term contained in title
        score = client._calculate_match_score("diabetes", "type 1 diabetes")
        assert score == 0.8
        
        # Word-based matching
        score = client._calculate_match_score("heart disease", "coronary heart condition")
        assert 0.0 < score < 0.8
        
        # No match
        score = client._calculate_match_score("diabetes", "hypertension")
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_process_search_results(self, client, mock_search_response):
        """Test processing of search results"""
        result = await client._process_search_results(mock_search_response, "diabetes", 10)
        
        assert result['success'] is True
        assert result['term'] == "diabetes"
        assert result['total_results'] == 2
        assert len(result['results']) == 2
        
        # Check first result
        first_result = result['results'][0]
        assert first_result['title'] == "Diabetes mellitus"
        assert first_result['code'] == "E10-E14"
        assert "Diabetes" in first_result['synonyms']
        assert first_result['rank'] == 1
    
    def test_extract_parent_categories(self, client):
        """Test extraction of parent categories"""
        entity_data = {
            "parent": [
                {"title": {"@value": "Endocrine disorders"}},
                {"title": {"@value": "Metabolic disorders"}}
            ]
        }
        
        parents = client._extract_parent_categories(entity_data)
        
        assert len(parents) == 2
        assert "Endocrine disorders" in parents
        assert "Metabolic disorders" in parents
    
    def test_extract_children(self, client):
        """Test extraction of child categories"""
        entity_data = {
            "child": [
                {"title": {"@value": "Type 1 diabetes"}},
                {"title": {"@value": "Type 2 diabetes"}},
                {"title": {"@value": "Gestational diabetes"}}
            ]
        }
        
        children = client._extract_children(entity_data)
        
        assert len(children) == 3
        assert "Type 1 diabetes" in children
        assert "Type 2 diabetes" in children
    
    def test_extract_additional_info(self, client):
        """Test extraction of additional information"""
        entity_data = {
            "inclusion": [
                {"@value": "Includes diabetes with complications"},
                {"@value": "Includes insulin-dependent diabetes"}
            ],
            "exclusion": [
                {"@value": "Excludes gestational diabetes"}
            ],
            "codingNote": [
                {"@value": "Use additional code for complications"}
            ]
        }
        
        info = client._extract_additional_info(entity_data)
        
        assert len(info['inclusions']) == 2
        assert len(info['exclusions']) == 1
        assert len(info['coding_notes']) == 1
        assert "complications" in info['inclusions'][0]
    
    def test_extract_text_array(self, client):
        """Test text array extraction from various formats"""
        # List of dictionaries
        data1 = [{"@value": "text1"}, {"@value": "text2"}]
        result1 = client._extract_text_array(data1)
        assert result1 == ["text1", "text2"]
        
        # Single dictionary
        data2 = {"@value": "single text"}
        result2 = client._extract_text_array(data2)
        assert result2 == ["single text"]
        
        # List of strings
        data3 = ["string1", "string2"]
        result3 = client._extract_text_array(data3)
        assert result3 == ["string1", "string2"]
        
        # Single string
        data4 = "single string"
        result4 = client._extract_text_array(data4)
        assert result4 == ["single string"]
    
    @pytest.mark.asyncio
    async def test_get_entity_details(self, client):
        """Test getting entity details"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "parent": [{"title": {"@value": "Parent category"}}],
            "child": [{"title": {"@value": "Child category"}}],
            "inclusion": [{"@value": "Inclusion note"}]
        }
        
        with patch.object(client.session, 'get', return_value=mock_response):
            details = await client._get_entity_details("123456")
        
        assert 'parent_categories' in details
        assert 'children' in details
        assert 'additional_info' in details
        assert "Parent category" in details['parent_categories']
    
    @pytest.mark.asyncio
    async def test_get_disease_hierarchy(self, client):
        """Test getting disease hierarchy"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "parent": [{"title": {"@value": "Metabolic disorders"}}],
            "child": [{"title": {"@value": "Type 1 diabetes"}}]
        }
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.get_disease_hierarchy("E10")
        
        assert result['success'] is True
        assert result['code'] == "E10"
        assert 'hierarchy' in result
        assert len(result['hierarchy']['parents']) == 1
        assert len(result['hierarchy']['children']) == 1
    
    def test_determine_hierarchy_level(self, client):
        """Test hierarchy level determination"""
        # Chapter level (no parent)
        data1 = {"child": [{"title": "child"}]}
        level1 = client._determine_hierarchy_level(data1)
        assert level1 == "chapter"
        
        # Category level (has parent and children)
        data2 = {"parent": [{"title": "parent"}], "child": [{"title": "child"}]}
        level2 = client._determine_hierarchy_level(data2)
        assert level2 == "category"
        
        # Subcategory level (has parent but no children)
        data3 = {"parent": [{"title": "parent"}]}
        level3 = client._determine_hierarchy_level(data3)
        assert level3 == "subcategory"
    
    @pytest.mark.asyncio
    async def test_validate_medical_code_valid(self, client):
        """Test validating a valid medical code"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": {"@value": "Diabetes mellitus"},
            "definition": {"@value": "A metabolic disorder"}
        }
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.validate_medical_code("E10")
        
        assert result['valid'] is True
        assert result['code'] == "E10"
        assert result['title'] == "Diabetes mellitus"
    
    @pytest.mark.asyncio
    async def test_validate_medical_code_invalid(self, client):
        """Test validating an invalid medical code"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.validate_medical_code("INVALID")
        
        assert result['valid'] is False
        assert result['code'] == "INVALID"
        assert 'Code not found' in result['error']
    
    def test_test_connection_success(self, client):
        """Test successful connection test"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.test_connection()
        
        assert result is True
    
    def test_test_connection_auth_reachable(self, client):
        """Test connection test when API is reachable but auth fails"""
        mock_response = Mock()
        mock_response.status_code = 401
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.test_connection()
        
        assert result is True  # API is reachable, just auth failed
    
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
        assert 'endpoints' in info
        assert info['api_key_configured'] is True
        assert 'search' in info['endpoints']
        assert 'entity' in info['endpoints']
    
    def test_close_client(self, client):
        """Test closing the client"""
        mock_session = Mock()
        client.session = mock_session
        
        client.close()
        
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_term_with_parameters(self, client, mock_search_response):
        """Test search with custom parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_response
        
        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            await client.search_term("diabetes", max_results=5, use_fuzzy_search=False)
        
        # Verify parameters were passed correctly
        call_args = mock_get.call_args
        params = call_args[1]['params']
        assert params['q'] == "diabetes"
        assert params['useFlexisearch'] == 'false'
    
    @pytest.mark.asyncio
    async def test_search_term_empty_results(self, client):
        """Test search with empty results"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"destinationEntities": []}
        
        with patch.object(client.session, 'get', return_value=mock_response):
            result = await client.search_term("nonexistent_term")
        
        assert result['success'] is True
        assert result['total_results'] == 0
        assert result['returned_results'] == 0
        assert result['results'] == []
        assert result['best_match'] is None