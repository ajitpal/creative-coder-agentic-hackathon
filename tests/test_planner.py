"""
Unit tests for planner module
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from src.planner import QueryPlanner, QueryPlan, UrgencyLevel, ConfidenceLevel
from src.models import MedicalQuery, QueryType

class TestQueryPlanner:
    """Test cases for QueryPlanner class"""
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Mock Gemini client for testing"""
        client = Mock()
        client.generate_response = AsyncMock()
        return client
    
    @pytest.fixture
    def planner(self, mock_gemini_client):
        """Create QueryPlanner instance for testing"""
        return QueryPlanner(gemini_client=mock_gemini_client)
    
    @pytest.fixture
    def planner_no_gemini(self):
        """Create QueryPlanner without Gemini client"""
        return QueryPlanner(gemini_client=None)
    
    @pytest.fixture
    def sample_query(self):
        """Create sample medical query"""
        return MedicalQuery(
            query_text="What is diabetes?",
            query_type=QueryType.MEDICAL_TERM
        )
    
    def test_planner_initialization(self, planner):
        """Test planner initialization"""
        assert planner.gemini_client is not None
        assert planner.classification_patterns is not None
        assert planner.available_tools is not None
        assert planner.emergency_keywords is not None
        assert planner.disclaimer_templates is not None
        
        # Check that patterns are loaded
        assert 'medical_term' in planner.classification_patterns
        assert 'drug_info' in planner.classification_patterns
        assert 'symptoms' in planner.classification_patterns
        
        # Check that tools are loaded
        assert 'search_medical_term' in planner.available_tools
        assert 'get_drug_information' in planner.available_tools
        assert 'analyze_symptoms' in planner.available_tools
    
    def test_classify_query_basic_medical_term(self, planner):
        """Test basic query classification for medical terms"""
        result = planner._classify_query_basic("What is diabetes?")
        
        assert result['query_type'] == 'medical_term'
        assert result['confidence'] > 0.0
        assert 'diabetes' in result['key_entities']
        assert result['method'] == 'pattern_matching'
    
    def test_classify_query_basic_drug_info(self, planner):
        """Test basic query classification for drug information"""
        result = planner._classify_query_basic("Tell me about aspirin side effects")
        
        assert result['query_type'] == 'drug_info'
        assert result['confidence'] > 0.0
        assert 'aspirin' in result['key_entities']
    
    def test_classify_query_basic_symptoms(self, planner):
        """Test basic query classification for symptoms"""
        result = planner._classify_query_basic("I have a severe headache and nausea")
        
        assert result['query_type'] == 'symptoms'
        assert result['confidence'] > 0.0
        assert any(entity in ['headache', 'nausea'] for entity in result['key_entities'])
    
    def test_classify_query_basic_document_summary(self, planner):
        """Test basic query classification for document summary"""
        result = planner._classify_query_basic("Please summarize this medical report")
        
        assert result['query_type'] == 'document_summary'
        assert result['confidence'] > 0.0
    
    def test_classify_query_basic_entity_extraction(self, planner):
        """Test basic query classification for entity extraction"""
        result = planner._classify_query_basic("Extract medications from this text")
        
        assert result['query_type'] == 'entity_extraction'
        assert result['confidence'] > 0.0
    
    def test_classify_query_basic_unknown(self, planner):
        """Test basic query classification for unknown query"""
        result = planner._classify_query_basic("Hello, how are you today?")
        
        assert result['query_type'] == 'medical_term'  # Default fallback
        assert result['confidence'] <= 0.5
    
    def test_extract_basic_entities(self, planner):
        """Test basic entity extraction"""
        entities = planner._extract_basic_entities("Patient has diabetes and takes insulin")
        
        assert 'diabetes' in entities
        assert 'insulin' in entities
    
    def test_extract_basic_entities_quoted(self, planner):
        """Test entity extraction with quoted terms"""
        entities = planner._extract_basic_entities('Patient has "type 2 diabetes" condition')
        
        assert 'type 2 diabetes' in entities
    
    def test_assess_urgency_emergency(self, planner):
        """Test urgency assessment for emergency situations"""
        analysis = {'requires_immediate_attention': False, 'urgency_level': 'low'}
        
        urgency = planner._assess_urgency("Patient having chest pain and difficulty breathing", analysis)
        
        assert urgency == UrgencyLevel.EMERGENCY
    
    def test_assess_urgency_high_from_analysis(self, planner):
        """Test urgency assessment from Gemini analysis"""
        analysis = {'requires_immediate_attention': True, 'urgency_level': 'high'}
        
        urgency = planner._assess_urgency("Patient feeling unwell", analysis)
        
        assert urgency == UrgencyLevel.HIGH
    
    def test_assess_urgency_low(self, planner):
        """Test urgency assessment for low urgency"""
        analysis = {'requires_immediate_attention': False, 'urgency_level': 'low'}
        
        urgency = planner._assess_urgency("What is the definition of hypertension?", analysis)
        
        assert urgency == UrgencyLevel.LOW
    
    def test_determine_tools_needed_medical_term(self, planner):
        """Test tool determination for medical term queries"""
        analysis = {'suggested_tools': []}
        
        tools = planner._determine_tools_needed('medical_term', analysis)
        
        assert 'search_medical_term' in tools
    
    def test_determine_tools_needed_drug_info(self, planner):
        """Test tool determination for drug info queries"""
        analysis = {'suggested_tools': []}
        
        tools = planner._determine_tools_needed('drug_info', analysis)
        
        assert 'get_drug_information' in tools
    
    def test_determine_tools_needed_with_suggestions(self, planner):
        """Test tool determination with Gemini suggestions"""
        analysis = {
            'suggested_tools': ['extract_medical_entities'],
            'complexity_assessment': 'complex'
        }
        
        tools = planner._determine_tools_needed('medical_term', analysis)
        
        assert 'search_medical_term' in tools
        assert 'extract_medical_entities' in tools
    
    def test_create_execution_steps(self, planner):
        """Test execution step creation"""
        tools = ['search_medical_term', 'extract_medical_entities']
        analysis = {'key_entities': ['diabetes'], 'medical_context': 'test context'}
        context = {}
        
        steps = planner._create_execution_steps(tools, analysis, context)
        
        assert len(steps) == 2
        assert steps[0]['step_number'] == 1
        assert steps[0]['tool_name'] == 'search_medical_term'
        assert steps[1]['step_number'] == 2
        assert steps[1]['tool_name'] == 'extract_medical_entities'
        
        # Check that steps have required fields
        for step in steps:
            assert 'description' in step
            assert 'estimated_time' in step
            assert 'complexity' in step
            assert 'input_data' in step
            assert 'expected_output' in step
    
    def test_select_disclaimers_general(self, planner):
        """Test disclaimer selection for general queries"""
        disclaimers = planner._select_disclaimers('medical_term', UrgencyLevel.LOW)
        
        assert len(disclaimers) >= 2
        assert any('educational purposes' in disclaimer for disclaimer in disclaimers)
        assert any('not provide medical diagnoses' in disclaimer for disclaimer in disclaimers)
    
    def test_select_disclaimers_symptoms(self, planner):
        """Test disclaimer selection for symptom queries"""
        disclaimers = planner._select_disclaimers('symptoms', UrgencyLevel.MEDIUM)
        
        assert any('does not constitute medical diagnosis' in disclaimer for disclaimer in disclaimers)
    
    def test_select_disclaimers_emergency(self, planner):
        """Test disclaimer selection for emergency situations"""
        disclaimers = planner._select_disclaimers('symptoms', UrgencyLevel.EMERGENCY)
        
        assert any('emergency' in disclaimer.lower() for disclaimer in disclaimers)
        assert any('911' in disclaimer for disclaimer in disclaimers)
    
    def test_calculate_priority_emergency(self, planner):
        """Test priority calculation for emergency"""
        priority = planner._calculate_priority(UrgencyLevel.EMERGENCY, 0.9)
        
        assert priority == 1  # Highest priority
    
    def test_calculate_priority_low_confidence(self, planner):
        """Test priority calculation with low confidence"""
        priority = planner._calculate_priority(UrgencyLevel.LOW, 0.3)
        
        assert priority >= 4  # Lower priority due to low confidence
    
    def test_determine_context_requirements(self, planner):
        """Test context requirements determination"""
        tools = ['analyze_symptoms']
        
        requirements = planner._determine_context_requirements(tools)
        
        assert requirements['conversation_history'] is True
        assert requirements['medical_history'] is True
        assert requirements['current_medications'] is True
    
    def test_determine_context_requirements_simple(self, planner):
        """Test context requirements for simple tools"""
        tools = ['search_medical_term']
        
        requirements = planner._determine_context_requirements(tools)
        
        assert requirements['conversation_history'] is False
        assert requirements['medical_history'] is False
    
    def test_estimate_complexity_simple(self, planner):
        """Test complexity estimation for simple tools"""
        tools = ['search_medical_term']
        
        complexity = planner._estimate_complexity(tools)
        
        assert complexity == 'simple'
    
    def test_estimate_complexity_complex(self, planner):
        """Test complexity estimation for complex tools"""
        tools = ['summarize_medical_document', 'extract_medical_entities']
        
        complexity = planner._estimate_complexity(tools)
        
        assert complexity == 'complex'
    
    def test_create_fallback_plan(self, planner, sample_query):
        """Test fallback plan creation"""
        plan = planner._create_fallback_plan(sample_query)
        
        assert isinstance(plan, QueryPlan)
        assert plan.query_id == sample_query.id
        assert plan.query_type == QueryType.MEDICAL_TERM
        assert plan.confidence == 0.3
        assert plan.urgency_level == UrgencyLevel.LOW
        assert len(plan.tools_required) > 0
        assert len(plan.medical_disclaimers) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_gemini_success(self, planner, sample_query):
        """Test Gemini analysis with successful response"""
        basic_classification = {
            'query_type': 'medical_term',
            'confidence': 0.7,
            'key_entities': ['diabetes'],
            'reasoning': 'Basic analysis'
        }
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = '''
        {
            "query_type": "medical_term",
            "confidence": 0.9,
            "key_entities": ["diabetes", "blood sugar"],
            "urgency_level": "low",
            "reasoning": "User asking for definition of diabetes",
            "suggested_tools": ["search_medical_term"],
            "medical_context": "Diabetes is a common metabolic condition",
            "requires_immediate_attention": false,
            "complexity_assessment": "simple"
        }
        '''
        
        planner.gemini_client.generate_response.return_value = mock_response
        
        result = await planner._analyze_with_gemini(sample_query, {}, basic_classification)
        
        assert result['query_type'] == 'medical_term'
        assert result['confidence'] == 0.9
        assert 'diabetes' in result['key_entities']
        assert 'blood sugar' in result['key_entities']
        assert result['method'] == 'gemini_enhanced'
    
    @pytest.mark.asyncio
    async def test_analyze_with_gemini_json_parse_error(self, planner, sample_query):
        """Test Gemini analysis with JSON parse error"""
        basic_classification = {
            'query_type': 'medical_term',
            'confidence': 0.7,
            'key_entities': ['diabetes'],
            'reasoning': 'Basic analysis'
        }
        
        # Mock Gemini response with invalid JSON
        mock_response = Mock()
        mock_response.text = "This is not valid JSON response"
        
        planner.gemini_client.generate_response.return_value = mock_response
        
        result = await planner._analyze_with_gemini(sample_query, {}, basic_classification)
        
        # Should fallback to basic classification
        assert result == basic_classification
    
    @pytest.mark.asyncio
    async def test_analyze_with_gemini_exception(self, planner, sample_query):
        """Test Gemini analysis with exception"""
        basic_classification = {
            'query_type': 'medical_term',
            'confidence': 0.7,
            'key_entities': ['diabetes'],
            'reasoning': 'Basic analysis'
        }
        
        # Mock Gemini to raise exception
        planner.gemini_client.generate_response.side_effect = Exception("API Error")
        
        result = await planner._analyze_with_gemini(sample_query, {}, basic_classification)
        
        # Should fallback to basic classification
        assert result == basic_classification
    
    @pytest.mark.asyncio
    async def test_analyze_query_full_workflow(self, planner, sample_query):
        """Test complete query analysis workflow"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = '''
        {
            "query_type": "medical_term",
            "confidence": 0.85,
            "key_entities": ["diabetes"],
            "urgency_level": "low",
            "reasoning": "User requesting definition of diabetes",
            "suggested_tools": ["search_medical_term"],
            "complexity_assessment": "simple"
        }
        '''
        
        planner.gemini_client.generate_response.return_value = mock_response
        
        context = {'conversation_history': []}
        plan = await planner.analyze_query(sample_query, context)
        
        assert isinstance(plan, QueryPlan)
        assert plan.query_id == sample_query.id
        assert plan.query_type == QueryType.MEDICAL_TERM
        assert plan.confidence == 0.85
        assert plan.urgency_level == UrgencyLevel.LOW
        assert len(plan.tools_required) > 0
        assert len(plan.execution_steps) > 0
        assert len(plan.medical_disclaimers) > 0
        assert 'diabetes' in plan.key_entities
    
    @pytest.mark.asyncio
    async def test_analyze_query_without_gemini(self, planner_no_gemini, sample_query):
        """Test query analysis without Gemini client"""
        plan = await planner_no_gemini.analyze_query(sample_query)
        
        assert isinstance(plan, QueryPlan)
        assert plan.query_id == sample_query.id
        assert plan.query_type == QueryType.MEDICAL_TERM
        assert plan.confidence > 0.0
        assert len(plan.tools_required) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_query_with_exception(self, planner, sample_query):
        """Test query analysis with exception handling"""
        # Mock Gemini to raise exception
        planner.gemini_client.generate_response.side_effect = Exception("Analysis failed")
        
        plan = await planner.analyze_query(sample_query)
        
        # Should return fallback plan
        assert isinstance(plan, QueryPlan)
        assert plan.confidence == 0.3  # Fallback confidence
        assert 'Fallback plan' in plan.reasoning
    
    def test_get_planning_statistics(self, planner):
        """Test getting planning statistics"""
        stats = planner.get_planning_statistics()
        
        assert 'available_query_types' in stats
        assert 'available_tools' in stats
        assert 'emergency_keywords' in stats
        assert 'disclaimer_templates' in stats
        assert 'gemini_enabled' in stats
        assert 'classification_patterns' in stats
        assert 'tool_names' in stats
        
        assert stats['gemini_enabled'] is True
        assert stats['available_query_types'] > 0
        assert stats['available_tools'] > 0
        assert len(stats['tool_names']) > 0


class TestQueryPlan:
    """Test cases for QueryPlan dataclass"""
    
    def test_query_plan_creation(self):
        """Test creating QueryPlan"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['search_medical_term'],
            execution_steps=[{'step': 1}],
            priority=2,
            urgency_level=UrgencyLevel.MEDIUM,
            confidence=0.8,
            reasoning="Test reasoning",
            key_entities=['diabetes'],
            medical_disclaimers=['Test disclaimer']
        )
        
        assert plan.query_id == "test-123"
        assert plan.query_type == QueryType.MEDICAL_TERM
        assert plan.tools_required == ['search_medical_term']
        assert plan.priority == 2
        assert plan.urgency_level == UrgencyLevel.MEDIUM
        assert plan.confidence == 0.8
        assert 'diabetes' in plan.key_entities
    
    def test_query_plan_to_dict(self):
        """Test QueryPlan serialization to dictionary"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.DRUG_INFO,
            tools_required=['get_drug_information'],
            urgency_level=UrgencyLevel.HIGH,
            confidence=0.9
        )
        
        plan_dict = plan.to_dict()
        
        assert plan_dict['query_id'] == "test-123"
        assert plan_dict['query_type'] == 'drug_info'
        assert plan_dict['tools_required'] == ['get_drug_information']
        assert plan_dict['urgency_level'] == 'high'
        assert plan_dict['confidence'] == 0.9
        assert 'execution_steps' in plan_dict
        assert 'medical_disclaimers' in plan_dict
    
    def test_query_plan_defaults(self):
        """Test QueryPlan with default values"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.SYMPTOMS
        )
        
        assert plan.tools_required == []
        assert plan.execution_steps == []
        assert plan.priority == 1
        assert plan.urgency_level == UrgencyLevel.LOW
        assert plan.confidence == 0.0
        assert plan.key_entities == []
        assert plan.medical_disclaimers == []f
rom src.planner import ExecutionPlanGenerator

class TestExecutionPlanGenerator:
    """Test cases for ExecutionPlanGenerator class"""
    
    @pytest.fixture
    def planner(self):
        """Create QueryPlanner for testing"""
        return QueryPlanner(gemini_client=None)
    
    @pytest.fixture
    def plan_generator(self, planner):
        """Create ExecutionPlanGenerator instance for testing"""
        return ExecutionPlanGenerator(planner)
    
    @pytest.fixture
    def sample_plan(self):
        """Create sample query plan for testing"""
        return QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['search_medical_term', 'extract_medical_entities'],
            execution_steps=[
                {
                    'step_number': 1,
                    'tool_name': 'search_medical_term',
                    'estimated_time': 2.0,
                    'requires_context': False
                },
                {
                    'step_number': 2,
                    'tool_name': 'extract_medical_entities',
                    'estimated_time': 1.5,
                    'requires_context': False
                }
            ],
            urgency_level=UrgencyLevel.LOW,
            confidence=0.8,
            medical_disclaimers=['General disclaimer']
        )
    
    def test_plan_generator_initialization(self, plan_generator):
        """Test plan generator initialization"""
        assert plan_generator.query_planner is not None
        assert plan_generator.optimization_strategies is not None
        assert plan_generator.validation_rules is not None
        
        # Check that strategies are loaded
        assert 'parallel_execution' in plan_generator.optimization_strategies
        assert 'tool_chaining' in plan_generator.optimization_strategies
        assert 'early_termination' in plan_generator.optimization_strategies
        
        # Check that validation rules are loaded
        assert 'tool_compatibility' in plan_generator.validation_rules
        assert 'resource_constraints' in plan_generator.validation_rules
        assert 'medical_safety' in plan_generator.validation_rules
    
    def test_validate_plan_success(self, plan_generator, sample_plan):
        """Test successful plan validation"""
        result = plan_generator._validate_plan(sample_plan)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['score'] > 80
    
    def test_validate_plan_incompatible_tools(self, plan_generator):
        """Test plan validation with incompatible tools"""
        # Create plan with incompatible tools
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.DOCUMENT_SUMMARY,
            tools_required=['summarize_medical_document', 'extract_medical_entities'],
            execution_steps=[],
            medical_disclaimers=['General disclaimer']
        )
        
        result = plan_generator._validate_plan(plan)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('Incompatible tools' in error for error in result['errors'])
    
    def test_validate_plan_emergency_time_limit(self, plan_generator):
        """Test plan validation for emergency time limits"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.SYMPTOMS,
            tools_required=['analyze_symptoms'],
            execution_steps=[
                {
                    'step_number': 1,
                    'tool_name': 'analyze_symptoms',
                    'estimated_time': 10.0  # Exceeds emergency limit
                }
            ],
            urgency_level=UrgencyLevel.EMERGENCY,
            medical_disclaimers=['General disclaimer']
        )
        
        result = plan_generator._validate_plan(plan)
        
        assert result['valid'] is False
        assert any('Emergency query exceeds response time' in error for error in result['errors'])
    
    def test_fix_plan_issues_incompatible_tools(self, plan_generator):
        """Test fixing incompatible tool issues"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.DOCUMENT_SUMMARY,
            tools_required=['summarize_medical_document', 'extract_medical_entities'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'summarize_medical_document'},
                {'step_number': 2, 'tool_name': 'extract_medical_entities'}
            ],
            medical_disclaimers=['General disclaimer']
        )
        
        errors = ['Incompatible tools: summarize_medical_document and extract_medical_entities']
        fixed_plan = plan_generator._fix_plan_issues(plan, errors)
        
        assert len(fixed_plan.tools_required) == 1
        assert len(fixed_plan.execution_steps) == 1
        assert '[Auto-fixed]' in fixed_plan.reasoning
    
    def test_fix_plan_issues_emergency_time(self, plan_generator):
        """Test fixing emergency time limit issues"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.SYMPTOMS,
            tools_required=['analyze_symptoms', 'search_medical_term'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'analyze_symptoms'},
                {'step_number': 2, 'tool_name': 'search_medical_term'}
            ],
            urgency_level=UrgencyLevel.EMERGENCY,
            medical_disclaimers=['General disclaimer']
        )
        
        errors = ['Emergency query exceeds response time limit (5.0s)']
        fixed_plan = plan_generator._fix_plan_issues(plan, errors)
        
        assert len(fixed_plan.tools_required) == 1
        assert len(fixed_plan.execution_steps) == 1
        assert fixed_plan.estimated_complexity == 'simple'
    
    def test_optimize_for_parallel_execution(self, plan_generator, sample_plan):
        """Test parallel execution optimization"""
        optimized_plan = plan_generator._optimize_for_parallel_execution(sample_plan)
        
        # Both tools should be marked for parallel execution
        parallel_steps = [step for step in optimized_plan.execution_steps if step.get('parallel_group')]
        assert len(parallel_steps) == 2
        
        # Both should have the same step number (parallel)
        assert parallel_steps[0]['step_number'] == parallel_steps[1]['step_number']
        
        # Should have parallel group metadata
        for step in parallel_steps:
            assert step['parallel_group'] is True
            assert step['parallel_group_size'] == 2
    
    def test_optimize_tool_chaining(self, plan_generator):
        """Test tool chaining optimization"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['extract_medical_entities', 'search_medical_term'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'extract_medical_entities'},
                {'step_number': 2, 'tool_name': 'search_medical_term'}
            ]
        )
        
        optimized_plan = plan_generator._optimize_tool_chaining(plan)
        
        # First step should be marked as chain start
        assert optimized_plan.execution_steps[0].get('chain_start') is True
        assert 'chain_name' in optimized_plan.execution_steps[0]
        
        # Second step should be marked as chain member
        assert optimized_plan.execution_steps[1].get('chain_member') is True
        assert optimized_plan.execution_steps[1].get('uses_output_from') == 'extract_medical_entities'
    
    def test_optimize_for_early_termination_emergency(self, plan_generator):
        """Test early termination optimization for emergencies"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.SYMPTOMS,
            tools_required=['analyze_symptoms', 'search_medical_term', 'extract_medical_entities'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'analyze_symptoms'},
                {'step_number': 2, 'tool_name': 'search_medical_term'},
                {'step_number': 3, 'tool_name': 'extract_medical_entities'}
            ],
            urgency_level=UrgencyLevel.EMERGENCY
        )
        
        optimized_plan = plan_generator._optimize_for_early_termination(plan)
        
        # Should keep only critical tools (analyze_symptoms)
        assert len(optimized_plan.tools_required) == 1
        assert 'analyze_symptoms' in optimized_plan.tools_required
        assert len(optimized_plan.execution_steps) == 1
        
        # Should have early termination enabled
        assert optimized_plan.execution_steps[0]['early_termination_enabled'] is True
        assert optimized_plan.execution_steps[0]['confidence_threshold'] == 0.7
    
    def test_optimize_for_early_termination_no_critical_tools(self, plan_generator):
        """Test early termination when no critical tools are present"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['search_medical_term', 'extract_medical_entities'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'search_medical_term'},
                {'step_number': 2, 'tool_name': 'extract_medical_entities'}
            ],
            urgency_level=UrgencyLevel.EMERGENCY
        )
        
        optimized_plan = plan_generator._optimize_for_early_termination(plan)
        
        # Should keep only first tool
        assert len(optimized_plan.tools_required) == 1
        assert optimized_plan.tools_required[0] == 'search_medical_term'
        assert len(optimized_plan.execution_steps) == 1
    
    def test_add_execution_metadata(self, plan_generator, sample_plan):
        """Test adding execution metadata"""
        # First optimize for parallel execution to get metadata
        optimized_plan = plan_generator._optimize_for_parallel_execution(sample_plan)
        
        # Then add metadata
        final_plan = plan_generator._add_execution_metadata(optimized_plan, {})
        
        assert 'total_estimated_time' in final_plan.context_requirements
        assert 'parallel_execution' in final_plan.context_requirements
        assert 'tool_chaining' in final_plan.context_requirements
        assert 'early_termination' in final_plan.context_requirements
        
        # Should have optimization info in reasoning
        if final_plan.context_requirements['parallel_execution']:
            assert 'parallel execution' in final_plan.reasoning
    
    @pytest.mark.asyncio
    async def test_generate_optimized_plan_success(self, plan_generator, sample_plan):
        """Test successful optimized plan generation"""
        context = {'conversation_history': []}
        
        optimized_plan = await plan_generator.generate_optimized_plan(
            sample_plan, 
            context, 
            ['time', 'accuracy']
        )
        
        assert isinstance(optimized_plan, QueryPlan)
        assert optimized_plan.query_id == sample_plan.query_id
        assert len(optimized_plan.execution_steps) > 0
        
        # Should have metadata added
        assert 'total_estimated_time' in optimized_plan.context_requirements
    
    @pytest.mark.asyncio
    async def test_generate_optimized_plan_validation_failure(self, plan_generator):
        """Test optimized plan generation with validation failure"""
        # Create invalid plan
        invalid_plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.DOCUMENT_SUMMARY,
            tools_required=['summarize_medical_document', 'extract_medical_entities'],
            execution_steps=[],
            medical_disclaimers=['General disclaimer']
        )
        
        optimized_plan = await plan_generator.generate_optimized_plan(invalid_plan)
        
        # Should return fixed plan
        assert isinstance(optimized_plan, QueryPlan)
        assert len(optimized_plan.tools_required) == 1  # Fixed incompatible tools
    
    @pytest.mark.asyncio
    async def test_generate_optimized_plan_exception_handling(self, plan_generator, sample_plan):
        """Test optimized plan generation with exception"""
        # Mock validation to raise exception
        with patch.object(plan_generator, '_validate_plan', side_effect=Exception("Validation error")):
            optimized_plan = await plan_generator.generate_optimized_plan(sample_plan)
        
        # Should return original plan
        assert optimized_plan == sample_plan
    
    def test_get_optimization_statistics(self, plan_generator):
        """Test getting optimization statistics"""
        stats = plan_generator.get_optimization_statistics()
        
        assert 'available_strategies' in stats
        assert 'validation_rules' in stats
        assert 'strategy_names' in stats
        assert 'validation_categories' in stats
        assert 'parallel_capable_tools' in stats
        assert 'tool_chains_available' in stats
        
        assert stats['available_strategies'] > 0
        assert stats['validation_rules'] > 0
        assert len(stats['strategy_names']) > 0