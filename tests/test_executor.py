"""
Unit tests for the ToolExecutor class
Tests tool execution coordination and result aggregation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

from src.executor import ToolExecutor, ExecutionResult
from src.planner import QueryPlan, UrgencyLevel
from src.models import QueryType, MedicalResponse, MedicalEntity, EntityType
from src.api_manager import APIManager


class TestToolExecutor:
    """Test cases for ToolExecutor class"""
    
    @pytest.fixture
    def mock_api_manager(self):
        """Create mock API manager"""
        api_manager = Mock(spec=APIManager)
        api_manager.who_icd_client = AsyncMock()
        api_manager.openfda_client = AsyncMock()
        api_manager.gemini_client = AsyncMock()
        return api_manager
    
    @pytest.fixture
    def tool_executor(self, mock_api_manager):
        """Create ToolExecutor instance with mocked dependencies"""
        with patch('src.executor.MedicalEntityExtractor'), \
             patch('src.executor.DocumentSummarizer'), \
             patch('src.executor.DocumentProcessor'):
            executor = ToolExecutor(api_manager=mock_api_manager)
            return executor
    
    @pytest.fixture
    def sample_query_plan(self):
        """Create sample query plan for testing"""
        return QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['search_medical_term'],
            execution_steps=[
                {
                    'step_number': 1,
                    'tool_name': 'search_medical_term',
                    'input_data': {'term': 'diabetes'},
                    'confidence_threshold': 0.7
                }
            ],
            priority=1,
            urgency_level=UrgencyLevel.LOW,
            confidence=0.8,
            key_entities=['diabetes'],
            reasoning="User wants to understand diabetes"
        )
    
    def test_tool_executor_initialization(self, tool_executor):
        """Test ToolExecutor initialization"""
        assert tool_executor is not None
        assert len(tool_executor.tools) == 5
        assert 'search_medical_term' in tool_executor.tools
        assert 'get_drug_information' in tool_executor.tools
        assert 'analyze_symptoms' in tool_executor.tools
        assert 'extract_medical_entities' in tool_executor.tools
        assert 'summarize_medical_document' in tool_executor.tools
        
        # Check tool metadata
        medical_term_tool = tool_executor.tools['search_medical_term']
        assert medical_term_tool['type'] == 'api_call'
        assert medical_term_tool['timeout'] == 10.0
        assert medical_term_tool['retry_count'] == 3
        
        # Check execution statistics initialization
        stats = tool_executor.execution_stats
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 0
        assert len(stats['tool_usage_count']) == 5
    
    @pytest.mark.asyncio
    async def test_execute_plan_success(self, tool_executor, sample_query_plan, mock_api_manager):
        """Test successful plan execution"""
        # Mock successful WHO ICD API response
        mock_api_manager.who_icd_client.search_medical_term.return_value = {
            'success': True,
            'data': {
                'definition': 'A group of metabolic disorders characterized by high blood sugar',
                'code': 'E10-E14',
                'category': 'Endocrine disorders'
            }
        }
        
        # Mock successful Gemini simplification
        mock_api_manager.gemini_client.generate_text.return_value = {
            'success': True,
            'text': 'Diabetes is a condition where your blood sugar is too high.'
        }
        
        result = await tool_executor.execute_plan(sample_query_plan)
        
        assert isinstance(result, MedicalResponse)
        assert result.query_id == "test-123"
        assert result.confidence_score > 0
        assert len(result.sources) > 0
        assert 'diabetes' in result.response_text.lower()
        
        # Check execution statistics were updated
        stats = tool_executor.execution_stats
        assert stats['total_executions'] == 1
        assert stats['successful_executions'] == 1
        assert stats['tool_usage_count']['search_medical_term'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_plan_with_failure(self, tool_executor, sample_query_plan, mock_api_manager):
        """Test plan execution with tool failure"""
        # Mock API failure
        mock_api_manager.who_icd_client.search_medical_term.side_effect = Exception("API Error")
        mock_api_manager.gemini_client.generate_text.side_effect = Exception("Gemini Error")
        
        result = await tool_executor.execute_plan(sample_query_plan)
        
        assert isinstance(result, MedicalResponse)
        assert result.confidence_score == 0.0
        assert 'error' in result.response_text.lower()
        
        # Check failure statistics were updated
        stats = tool_executor.execution_stats
        assert stats['failed_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_failure(self, tool_executor):
        """Test retry mechanism with eventual success"""
        mock_func = AsyncMock()
        mock_func.side_effect = [Exception("First failure"), Exception("Second failure"), {'success': True}]
        
        result = await tool_executor._execute_with_retry(mock_func, {}, max_retries=3, timeout=5.0)
        
        assert result == {'success': True}
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_timeout(self, tool_executor):
        """Test retry mechanism with timeout"""
        async def slow_function(data):
            await asyncio.sleep(2.0)
            return {'success': True}
        
        with pytest.raises(asyncio.TimeoutError):
            await tool_executor._execute_with_retry(slow_function, {}, max_retries=1, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_execute_tool_by_type(self, tool_executor, mock_api_manager):
        """Test executing tools by type"""
        # Mock API responses
        mock_api_manager.who_icd_client.search_medical_term.return_value = {
            'success': True,
            'data': {'definition': 'Test definition'},
            'confidence_score': 0.8
        }
        mock_api_manager.openfda_client.search_drug_comprehensive.return_value = {
            'recalls': {},
            'adverse_events': {},
            'label_info': {},
            'overall_safety_assessment': {},
            'confidence_score': 0.7
        }
        
        results = await tool_executor.execute_tool_by_type('api_call', {'term': 'test'})
        
        assert len(results) == 2  # search_medical_term and get_drug_information
        assert all(isinstance(r, ExecutionResult) for r in results)
        api_call_results = [r for r in results if r.success]
        assert len(api_call_results) >= 1
    
    @pytest.mark.asyncio
    async def test_execute_parallel_tools(self, tool_executor, mock_api_manager):
        """Test parallel tool execution"""
        # Mock API responses
        mock_api_manager.who_icd_client.search_medical_term.return_value = {
            'success': True,
            'data': {'definition': 'Test definition'},
            'confidence_score': 0.8
        }
        mock_api_manager.gemini_client.generate_text.return_value = {
            'success': True,
            'text': 'Test analysis',
            'confidence_score': 0.7
        }
        
        tool_names = ['search_medical_term', 'analyze_symptoms']
        input_data = {'term': 'test', 'symptoms': 'headache'}
        
        results = await tool_executor.execute_parallel_tools(tool_names, input_data)
        
        assert len(results) == 2
        assert all(isinstance(r, ExecutionResult) for r in results)
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 1
    
    @pytest.mark.asyncio
    async def test_medical_term_lookup_success(self, tool_executor, mock_api_manager):
        """Test successful medical term lookup"""
        mock_api_manager.who_icd_client.search_medical_term.return_value = {
            'success': True,
            'data': {
                'definition': 'A chronic condition affecting blood sugar',
                'code': 'E11',
                'category': 'Diabetes'
            }
        }
        mock_api_manager.gemini_client.generate_text.return_value = {
            'success': True,
            'text': 'Diabetes is when your body has trouble controlling blood sugar levels.'
        }
        
        input_data = {'query_entities': ['diabetes']}
        result = await tool_executor.execute_medical_term_lookup(input_data)
        
        assert result['term'] == 'diabetes'
        assert 'definition' in result
        assert result['source'] == 'WHO_ICD_with_AI_simplification'
        assert result['confidence_score'] == 0.9
    
    @pytest.mark.asyncio
    async def test_medical_term_lookup_fallback(self, tool_executor, mock_api_manager):
        """Test medical term lookup with fallback to Gemini"""
        mock_api_manager.who_icd_client.search_medical_term.return_value = {
            'success': False,
            'data': None
        }
        mock_api_manager.gemini_client.generate_text.return_value = {
            'success': True,
            'text': 'Hypertension is high blood pressure that can lead to serious health problems.'
        }
        
        input_data = {'query_entities': ['hypertension']}
        result = await tool_executor.execute_medical_term_lookup(input_data)
        
        assert result['term'] == 'hypertension'
        assert result['source'] == 'AI_generated_explanation'
        assert result['fallback_used'] is True
        assert result['confidence_score'] == 0.7
    
    @pytest.mark.asyncio
    async def test_drug_info_lookup_comprehensive(self, tool_executor, mock_api_manager):
        """Test comprehensive drug information lookup"""
        mock_api_manager.openfda_client.search_drug_comprehensive.return_value = {
            'recalls': {'total_recalls': 2, 'recalls': []},
            'adverse_events': {'total_events': 150, 'summary': {}},
            'label_info': {
                'brand_names': ['Aspirin'],
                'generic_names': ['acetylsalicylic acid'],
                'manufacturer': ['Bayer']
            },
            'overall_safety_assessment': {
                'safety_level': 'moderate',
                'concerns': ['stomach irritation'],
                'recommendations': ['take with food']
            }
        }
        
        input_data = {'query_entities': ['aspirin']}
        result = await tool_executor.execute_drug_info_lookup(input_data)
        
        assert result['drug_name'] == 'aspirin'
        assert result['source'] == 'OpenFDA_comprehensive'
        assert 'basic_info' in result
        assert 'safety_profile' in result
        assert 'recalls' in result
        assert result['confidence_score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_symptom_analysis_success(self, tool_executor, mock_api_manager):
        """Test successful symptom analysis"""
        mock_api_manager.gemini_client.generate_text.return_value = {
            'success': True,
            'text': '''**Possible Causes**: Headaches can be caused by stress, dehydration, or tension.
            **Severity Assessment**: This appears to be routine.
            **Self-Care**: Rest, hydration, and over-the-counter pain relief.
            **When to Seek Care**: If headaches persist or worsen.
            **Red Flags**: Sudden severe headache or vision changes.'''
        }
        
        input_data = {'query_entities': ['headache'], 'context': {}}
        result = await tool_executor.execute_symptom_analysis(input_data)
        
        assert result['symptoms_analyzed'] == 'headache'
        assert 'analysis' in result
        assert result['urgency_level'] in ['low', 'moderate', 'high']
        assert 'medical_disclaimers' in result
        assert result['confidence_score'] == 0.75
    
    @pytest.mark.asyncio
    async def test_entity_extraction_success(self, tool_executor):
        """Test successful entity extraction"""
        input_data = {'query_entities': ['diabetes', 'insulin', 'headache']}
        result = await tool_executor.execute_entity_extraction(input_data)
        
        assert 'entities_found' in result
        assert result['total_entities'] == 3
        assert result['confidence_score'] == 0.75
        
        # Check entity categorization
        entities = result['entities_found']
        diabetes_entity = next((e for e in entities if e['text'] == 'diabetes'), None)
        assert diabetes_entity is not None
        assert diabetes_entity['entity_type'] == 'disease'
        
        insulin_entity = next((e for e in entities if e['text'] == 'insulin'), None)
        assert insulin_entity is not None
        assert insulin_entity['entity_type'] == 'drug'
    
    @pytest.mark.asyncio
    async def test_document_summary_success(self, tool_executor):
        """Test successful document summarization"""
        with patch.object(tool_executor.document_processor, 'process_and_summarize_document') as mock_processor:
            mock_processor.return_value = {
                'success': True,
                'summary_info': {
                    'document_type': 'prescription',
                    'summary': 'Patient prescribed medication X for condition Y',
                    'key_findings': ['Medication X', 'Condition Y'],
                    'extracted_entities': [{'text': 'Medication X', 'type': 'drug'}],
                    'confidence_score': 0.8
                },
                'document_info': {'processing_time': 2.5}
            }
            
            input_data = {
                'document_text': 'Patient John Doe is prescribed medication X for condition Y...',
                'file_name': 'prescription.pdf'
            }
            result = await tool_executor.execute_document_summary(input_data)
            
            assert result['document_name'] == 'prescription.pdf'
            assert result['document_type'] == 'prescription'
            assert 'summary' in result
            assert 'key_findings' in result
            assert result['confidence_score'] == 0.8
    
    def test_get_execution_statistics(self, tool_executor):
        """Test getting execution statistics"""
        stats = tool_executor.get_execution_statistics()
        
        assert 'available_tools' in stats
        assert 'tool_names' in stats
        assert 'execution_stats' in stats
        assert stats['available_tools'] == 5
        assert len(stats['tool_names']) == 5
    
    def test_reset_statistics(self, tool_executor):
        """Test resetting execution statistics"""
        # Simulate some executions
        tool_executor.execution_stats['total_executions'] = 10
        tool_executor.execution_stats['successful_executions'] = 8
        
        tool_executor.reset_statistics()
        
        stats = tool_executor.execution_stats
        assert stats['total_executions'] == 0
        assert stats['successful_executions'] == 0
        assert stats['failed_executions'] == 0
    
    def test_get_tool_info(self, tool_executor):
        """Test getting information about specific tools"""
        info = tool_executor.get_tool_info('search_medical_term')
        
        assert info['name'] == 'search_medical_term'
        assert info['type'] == 'api_call'
        assert 'description' in info
        assert 'timeout' in info
        assert 'retry_count' in info
        
        # Test non-existent tool
        error_info = tool_executor.get_tool_info('non_existent_tool')
        assert 'error' in error_info
    
    def test_get_tools_by_type(self, tool_executor):
        """Test getting tools by type"""
        api_tools = tool_executor.get_tools_by_type('api_call')
        nlp_tools = tool_executor.get_tools_by_type('nlp_processing')
        doc_tools = tool_executor.get_tools_by_type('document_handling')
        
        assert 'search_medical_term' in api_tools
        assert 'get_drug_information' in api_tools
        assert 'analyze_symptoms' in nlp_tools
        assert 'extract_medical_entities' in nlp_tools
        assert 'summarize_medical_document' in doc_tools
    
    @pytest.mark.asyncio
    async def test_validate_tool_inputs(self, tool_executor):
        """Test tool input validation"""
        # Valid input for API call
        valid_result = await tool_executor.validate_tool_inputs(
            'search_medical_term', 
            {'term': 'diabetes'}
        )
        assert valid_result['valid'] is True
        
        # Invalid input for API call
        invalid_result = await tool_executor.validate_tool_inputs(
            'search_medical_term', 
            {}
        )
        assert invalid_result['valid'] is False
        assert 'error' in invalid_result
        
        # Valid input for NLP processing
        nlp_valid = await tool_executor.validate_tool_inputs(
            'analyze_symptoms',
            {'symptoms': 'headache'}
        )
        assert nlp_valid['valid'] is True
        
        # Valid input for document handling
        doc_valid = await tool_executor.validate_tool_inputs(
            'summarize_medical_document',
            {'document_text': 'Patient report...'}
        )
        assert doc_valid['valid'] is True
    
    def test_update_average_execution_time(self, tool_executor):
        """Test average execution time calculation"""
        # First execution
        tool_executor.execution_stats['total_executions'] = 1
        tool_executor._update_average_execution_time(2.0)
        assert tool_executor.execution_stats['average_execution_time'] == 2.0
        
        # Second execution
        tool_executor.execution_stats['total_executions'] = 2
        tool_executor._update_average_execution_time(4.0)
        assert tool_executor.execution_stats['average_execution_time'] == 3.0
        
        # Third execution
        tool_executor.execution_stats['total_executions'] = 3
        tool_executor._update_average_execution_time(3.0)
        assert tool_executor.execution_stats['average_execution_time'] == 3.0


class TestExecutionResult:
    """Test cases for ExecutionResult dataclass"""
    
    def test_execution_result_creation(self):
        """Test ExecutionResult creation"""
        result = ExecutionResult(
            success=True,
            data={'test': 'data'},
            tool_name='test_tool',
            execution_time=1.5,
            confidence_score=0.8,
            metadata={'step': 1}
        )
        
        assert result.success is True
        assert result.data == {'test': 'data'}
        assert result.tool_name == 'test_tool'
        assert result.execution_time == 1.5
        assert result.confidence_score == 0.8
        assert result.error is None
        assert result.metadata == {'step': 1}
    
    def test_execution_result_with_error(self):
        """Test ExecutionResult with error"""
        result = ExecutionResult(
            success=False,
            data={},
            tool_name='failing_tool',
            execution_time=0.5,
            confidence_score=0.0,
            error='Tool execution failed'
        )
        
        assert result.success is False
        assert result.error == 'Tool execution failed'
        assert result.confidence_score == 0.0


if __name__ == '__main__':
    pytest.main([__file__])