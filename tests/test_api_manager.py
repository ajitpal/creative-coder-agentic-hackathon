"""
Unit tests for API Manager
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
from src.api_manager import APIManager, APIResult, APIStatus
from src.memory import CacheManager

class TestAPIManager:
    """Test cases for APIManager class"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing"""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get_cached_response.return_value = None
        cache_manager.cache_api_response.return_value = "cache_key"
        cache_manager.get_cache_stats.return_value = {'total_entries': 0}
        cache_manager.close.return_value = None
        return cache_manager
    
    @pytest.fixture
    def api_manager(self, mock_cache_manager):
        """Create APIManager instance for testing"""
        with patch('src.api_manager.GeminiAPIClient') as mock_gemini, \
             patch('src.api_manager.WHOICDClient') as mock_who, \
             patch('src.api_manager.OpenFDAClient') as mock_fda:
            
            # Mock successful client initialization
            mock_gemini.return_value = Mock()
            mock_who.return_value = Mock()
            mock_fda.return_value = Mock()
            
            manager = APIManager(cache_manager=mock_cache_manager)
            return manager
    
    def test_api_manager_initialization(self, mock_cache_manager):
        """Test API manager initialization"""
        with patch('src.api_manager.GeminiAPIClient') as mock_gemini, \
             patch('src.api_manager.WHOICDClient') as mock_who, \
             patch('src.api_manager.OpenFDAClient') as mock_fda, \
             patch('src.config.Config.GEMINI_API_KEY', 'test_key'):
            
            mock_gemini.return_value = Mock()
            mock_who.return_value = Mock()
            mock_fda.return_value = Mock()
            
            manager = APIManager(cache_manager=mock_cache_manager)
            
            assert manager.cache_manager == mock_cache_manager
            assert manager.gemini_client is not None
            assert manager.who_icd_client is not None
            assert manager.openfda_client is not None
            assert len(manager.api_status) == 3
    
    def test_api_manager_initialization_without_gemini_key(self, mock_cache_manager):
        """Test API manager initialization without Gemini API key"""
        with patch('src.api_manager.GeminiAPIClient') as mock_gemini, \
             patch('src.api_manager.WHOICDClient') as mock_who, \
             patch('src.api_manager.OpenFDAClient') as mock_fda, \
             patch('src.config.Config.GEMINI_API_KEY', None):
            
            mock_who.return_value = Mock()
            mock_fda.return_value = Mock()
            
            manager = APIManager(cache_manager=mock_cache_manager)
            
            assert manager.gemini_client is None
            assert manager.api_status['gemini'] == APIStatus.UNAVAILABLE
            mock_gemini.assert_not_called()
    
    def test_api_manager_initialization_with_client_error(self, mock_cache_manager):
        """Test API manager initialization with client initialization error"""
        with patch('src.api_manager.GeminiAPIClient', side_effect=Exception("Init error")), \
             patch('src.api_manager.WHOICDClient') as mock_who, \
             patch('src.api_manager.OpenFDAClient') as mock_fda, \
             patch('src.config.Config.GEMINI_API_KEY', 'test_key'):
            
            mock_who.return_value = Mock()
            mock_fda.return_value = Mock()
            
            manager = APIManager(cache_manager=mock_cache_manager)
            
            assert manager.gemini_client is None
            assert manager.api_status['gemini'] == APIStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_check_api_status(self, api_manager):
        """Test API status checking"""
        # Mock test_connection methods
        api_manager.gemini_client.test_connection.return_value = True
        api_manager.who_icd_client.test_connection.return_value = True
        api_manager.openfda_client.test_connection.return_value = False
        
        status = await api_manager.check_api_status()
        
        assert status['gemini'] == APIStatus.AVAILABLE
        assert status['who_icd'] == APIStatus.AVAILABLE
        assert status['openfda'] == APIStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_check_api_status_single_api(self, api_manager):
        """Test checking status of single API"""
        api_manager.gemini_client.test_connection.return_value = True
        
        status = await api_manager.check_api_status('gemini')
        
        assert status['gemini'] == APIStatus.AVAILABLE
        # Other APIs should not be checked
        api_manager.who_icd_client.test_connection.assert_not_called()
        api_manager.openfda_client.test_connection.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_check_api_status_with_exception(self, api_manager):
        """Test API status checking with exception"""
        api_manager.gemini_client.test_connection.side_effect = Exception("Connection error")
        
        status = await api_manager.check_api_status('gemini')
        
        assert status['gemini'] == APIStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, api_manager):
        """Test successful execution with retry logic"""
        mock_func = AsyncMock(return_value={'success': True, 'data': 'test'})
        
        result = await api_manager._execute_with_retry(mock_func, 'test_api', 'cache_key')
        
        assert result.success is True
        assert result.data['success'] is True
        assert result.source == 'test_api'
        assert result.retry_count == 0
        mock_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_from_cache(self, api_manager):
        """Test execution with cached result"""
        cached_data = {'success': True, 'cached': True}
        api_manager.cache_manager.get_cached_response.return_value = cached_data
        
        mock_func = AsyncMock()
        
        result = await api_manager._execute_with_retry(mock_func, 'test_api', 'cache_key')
        
        assert result.success is True
        assert result.cached is True
        assert result.data == cached_data
        mock_func.assert_not_called()  # Should not call function if cached
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_with_retries(self, api_manager):
        """Test execution with retries on failure"""
        mock_func = AsyncMock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            {'success': True, 'data': 'success_after_retries'}
        ])
        
        result = await api_manager._execute_with_retry(mock_func, 'test_api')
        
        assert result.success is True
        assert result.retry_count == 2
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_all_failures(self, api_manager):
        """Test execution when all retries fail"""
        mock_func = AsyncMock(side_effect=Exception("Persistent failure"))
        
        result = await api_manager._execute_with_retry(mock_func, 'test_api')
        
        assert result.success is False
        assert result.retry_count == api_manager.max_retries
        assert "Persistent failure" in result.error
        assert mock_func.call_count == api_manager.max_retries + 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_unavailable_api(self, api_manager):
        """Test execution when API is unavailable"""
        api_manager.api_status['test_api'] = APIStatus.UNAVAILABLE
        mock_func = AsyncMock()
        
        result = await api_manager._execute_with_retry(mock_func, 'test_api')
        
        assert result.success is False
        assert "unavailable" in result.error
        mock_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_medical_term_success(self, api_manager):
        """Test successful medical term search"""
        mock_search_result = {
            'success': True,
            'term': 'diabetes',
            'definition': 'A metabolic disorder'
        }
        
        api_manager.who_icd_client.search_term = AsyncMock(return_value=mock_search_result)
        
        result = await api_manager.search_medical_term('diabetes')
        
        assert result.success is True
        assert result.data['term'] == 'diabetes'
        assert result.source == 'who_icd'
    
    @pytest.mark.asyncio
    async def test_search_medical_term_with_fallback(self, api_manager):
        """Test medical term search with Gemini fallback"""
        # WHO ICD fails
        api_manager.who_icd_client.search_term = AsyncMock(return_value={'success': False})
        
        # Gemini succeeds
        mock_gemini_response = {
            'text': 'Diabetes is a metabolic disorder...',
            'success': True
        }
        api_manager.gemini_client.generate_response = AsyncMock(return_value=mock_gemini_response)
        
        result = await api_manager.search_medical_term('diabetes')
        
        assert result.success is True
        assert result.source == 'gemini_fallback'
        assert result.data['fallback_used'] is True
    
    @pytest.mark.asyncio
    async def test_search_medical_term_all_fail(self, api_manager):
        """Test medical term search when all sources fail"""
        # WHO ICD fails
        api_manager.who_icd_client.search_term = AsyncMock(return_value={'success': False})
        
        # Gemini fails
        api_manager.gemini_client.generate_response = AsyncMock(side_effect=Exception("Gemini error"))
        
        result = await api_manager.search_medical_term('diabetes')
        
        assert result.success is False
        assert result.source == 'none'
        assert 'All medical term lookup sources failed' in result.data['error']
    
    @pytest.mark.asyncio
    async def test_get_drug_information_success(self, api_manager):
        """Test successful drug information retrieval"""
        mock_drug_result = {
            'drug_name': 'aspirin',
            'recalls': [],
            'adverse_events': [],
            'success': True
        }
        
        api_manager.openfda_client.search_drug_comprehensive = AsyncMock(return_value=mock_drug_result)
        
        result = await api_manager.get_drug_information('aspirin')
        
        assert result.success is True
        assert result.data['drug_name'] == 'aspirin'
        assert result.source == 'openfda'
    
    @pytest.mark.asyncio
    async def test_get_drug_information_with_partial_fallback(self, api_manager):
        """Test drug information with partial fallback"""
        # Comprehensive search fails
        api_manager.openfda_client.search_drug_comprehensive = AsyncMock(return_value={'success': False})
        
        # Individual endpoints succeed
        mock_recalls = APIResult(True, {'recalls': []}, 'openfda_recalls', 0.5)
        mock_events = APIResult(True, {'adverse_events': []}, 'openfda_events', 0.3)
        
        with patch.object(api_manager, '_execute_with_retry', side_effect=[mock_recalls, mock_events]):
            result = await api_manager.get_drug_information('aspirin')
        
        assert result.success is True
        assert result.data['partial_results'] is True
        assert result.source == 'openfda_partial'
    
    @pytest.mark.asyncio
    async def test_analyze_symptoms_with_gemini(self, api_manager):
        """Test symptom analysis with Gemini"""
        mock_analysis = {
            'text': 'These symptoms suggest...',
            'success': True
        }
        
        api_manager.gemini_client.generate_response = AsyncMock(return_value=mock_analysis)
        
        result = await api_manager.analyze_symptoms_with_gemini('headache and fever')
        
        assert result.success is True
        api_manager.gemini_client.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_symptoms_without_gemini(self, api_manager):
        """Test symptom analysis without Gemini client"""
        api_manager.gemini_client = None
        
        result = await api_manager.analyze_symptoms_with_gemini('headache')
        
        assert result.success is False
        assert 'Gemini API not available' in result.data['error']
    
    @pytest.mark.asyncio
    async def test_extract_medical_entities(self, api_manager):
        """Test medical entity extraction"""
        mock_response = {
            'text': '{"entities": [{"text": "diabetes", "type": "disease", "confidence": 0.9}]}',
            'success': True
        }
        
        api_manager.gemini_client.generate_response = AsyncMock(return_value=mock_response)
        
        result = await api_manager.extract_medical_entities('Patient has diabetes')
        
        assert result.success is True
        assert 'parsed_entities' in result.data
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_health_info_medical_term(self, api_manager):
        """Test comprehensive health info for medical term"""
        mock_term_result = APIResult(True, {'term': 'diabetes'}, 'who_icd', 0.5)
        
        with patch.object(api_manager, 'search_medical_term', return_value=mock_term_result):
            results = await api_manager.get_comprehensive_health_info('diabetes', 'medical_term')
        
        assert 'medical_term' in results
        assert results['medical_term'].success is True
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_health_info_drug(self, api_manager):
        """Test comprehensive health info for drug"""
        mock_drug_result = APIResult(True, {'drug_name': 'aspirin'}, 'openfda', 0.5)
        
        with patch.object(api_manager, 'get_drug_information', return_value=mock_drug_result):
            results = await api_manager.get_comprehensive_health_info('aspirin', 'drug_info')
        
        assert 'drug_info' in results
        assert results['drug_info'].success is True
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_health_info_general_query(self, api_manager):
        """Test comprehensive health info for general query"""
        # Mock query analysis
        mock_analysis = APIResult(
            True, 
            {'suggested_functions': ['search_medical_term', 'get_drug_information']}, 
            'gemini', 
            0.3
        )
        
        mock_term_result = APIResult(True, {'term': 'test'}, 'who_icd', 0.5)
        mock_drug_result = APIResult(True, {'drug_name': 'test'}, 'openfda', 0.4)
        
        with patch.object(api_manager, '_execute_with_retry', return_value=mock_analysis), \
             patch.object(api_manager, 'search_medical_term', return_value=mock_term_result), \
             patch.object(api_manager, 'get_drug_information', return_value=mock_drug_result):
            
            results = await api_manager.get_comprehensive_health_info('test query', 'general')
        
        assert 'query_analysis' in results
        assert 'medical_term' in results
        assert 'drug_info' in results
    
    def test_get_api_statistics(self, api_manager):
        """Test getting API statistics"""
        stats = api_manager.get_api_statistics()
        
        assert 'api_status' in stats
        assert 'cache_statistics' in stats
        assert 'clients_initialized' in stats
        assert 'retry_configuration' in stats
        
        assert stats['clients_initialized']['gemini'] is True
        assert stats['clients_initialized']['who_icd'] is True
        assert stats['clients_initialized']['openfda'] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_manager):
        """Test comprehensive health check"""
        # Mock API status checks
        api_manager.gemini_client.test_connection.return_value = True
        api_manager.who_icd_client.test_connection.return_value = True
        api_manager.openfda_client.test_connection.return_value = False
        
        health = await api_manager.health_check()
        
        assert 'overall' in health
        assert health['overall']['available_apis'] == 2
        assert health['overall']['total_apis'] == 3
        assert health['overall']['health_percentage'] == 66.7
        assert health['overall']['healthy'] is True
    
    @pytest.mark.asyncio
    async def test_health_check_all_apis_down(self, api_manager):
        """Test health check when all APIs are down"""
        # Mock all APIs as failing
        api_manager.gemini_client.test_connection.return_value = False
        api_manager.who_icd_client.test_connection.return_value = False
        api_manager.openfda_client.test_connection.return_value = False
        
        health = await api_manager.health_check()
        
        assert health['overall']['available_apis'] == 0
        assert health['overall']['health_percentage'] == 0.0
        assert health['overall']['healthy'] is False
    
    def test_close(self, api_manager):
        """Test closing API manager"""
        # Mock close methods
        api_manager.who_icd_client.close = Mock()
        api_manager.openfda_client.close = Mock()
        api_manager.cache_manager.close = Mock()
        
        api_manager.close()
        
        api_manager.who_icd_client.close.assert_called_once()
        api_manager.openfda_client.close.assert_called_once()
        api_manager.cache_manager.close.assert_called_once()
    
    def test_close_with_exception(self, api_manager):
        """Test closing API manager with exception"""
        api_manager.who_icd_client.close = Mock(side_effect=Exception("Close error"))
        
        # Should not raise exception
        api_manager.close()


class TestAPIResult:
    """Test cases for APIResult dataclass"""
    
    def test_api_result_creation(self):
        """Test creating APIResult"""
        result = APIResult(
            success=True,
            data={'test': 'data'},
            source='test_api',
            response_time=0.5,
            cached=False,
            error=None,
            retry_count=0
        )
        
        assert result.success is True
        assert result.data == {'test': 'data'}
        assert result.source == 'test_api'
        assert result.response_time == 0.5
        assert result.cached is False
        assert result.error is None
        assert result.retry_count == 0
    
    def test_api_result_with_error(self):
        """Test creating APIResult with error"""
        result = APIResult(
            success=False,
            data={},
            source='test_api',
            response_time=1.0,
            error='Test error',
            retry_count=3
        )
        
        assert result.success is False
        assert result.error == 'Test error'
        assert result.retry_count == 3