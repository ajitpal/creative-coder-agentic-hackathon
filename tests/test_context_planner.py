"""
Unit tests for context-aware planner
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.context_planner import ContextAwarePlanner
from src.planner import QueryPlanner, QueryPlan, UrgencyLevel
from src.models import MedicalQuery, QueryType

class TestContextAwarePlanner:
    """Test cases for ContextAwarePlanner class"""
    
    @pytest.fixture
    def mock_query_planner(self):
        """Mock query planner for testing"""
        planner = Mock(spec=QueryPlanner)
        planner.analyze_query = AsyncMock()
        return planner
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing"""
        return Mock()
    
    @pytest.fixture
    def context_planner(self, mock_query_planner, mock_memory_manager):
        """Create ContextAwarePlanner instance for testing"""
        return ContextAwarePlanner(mock_query_planner, mock_memory_manager)
    
    @pytest.fixture
    def sample_query(self):
        """Create sample medical query"""
        return MedicalQuery(
            query_text="Tell me more about diabetes complications",
            query_type=QueryType.MEDICAL_TERM
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context with conversation history and preferences"""
        return {
            'conversation_history': [
                {
                    'query_text': 'What is diabetes?',
                    'response_text': 'Diabetes is a metabolic disorder characterized by high blood sugar levels.',
                    'query_type': 'medical_term',
                    'medical_entities': [{'text': 'diabetes', 'type': 'disease'}]
                },
                {
                    'query_text': 'What are the symptoms?',
                    'response_text': 'Common symptoms include increased thirst, frequent urination, and fatigue.',
                    'query_type': 'symptoms',
                    'medical_entities': [{'text': 'thirst', 'type': 'symptom'}]
                }
            ],
            'user_preferences': {
                'allergies': ['penicillin', 'sulfa'],
                'language': 'english',
                'explanation_level': 'detailed'
            }
        }
    
    @pytest.fixture
    def sample_base_plan(self):
        """Create sample base plan"""
        return QueryPlan(
            query_id="test-123",
            query_type=QueryType.MEDICAL_TERM,
            tools_required=['search_medical_term'],
            execution_steps=[
                {
                    'step_number': 1,
                    'tool_name': 'search_medical_term',
                    'estimated_time': 2.0
                }
            ],
            confidence=0.7,
            key_entities=['diabetes'],
            medical_disclaimers=['General disclaimer']
        )
    
    def test_context_planner_initialization(self, context_planner):
        """Test context planner initialization"""
        assert context_planner.query_planner is not None
        assert context_planner.memory_manager is not None
        assert context_planner.context_patterns is not None
        assert context_planner.relevance_weights is not None
        assert context_planner.integration_strategies is not None
        
        # Check that patterns are loaded
        assert 'follow_up_queries' in context_planner.context_patterns
        assert 'clarification_requests' in context_planner.context_patterns
        assert 'preference_indicators' in context_planner.context_patterns
        
        # Check that strategies are loaded
        assert 'entity_enrichment' in context_planner.integration_strategies
        assert 'preference_filtering' in context_planner.integration_strategies
        assert 'conversation_threading' in context_planner.integration_strategies
    
    def test_detect_query_patterns_follow_up(self, context_planner):
        """Test detection of follow-up query patterns"""
        pattern = context_planner._detect_query_patterns("Tell me more about diabetes")
        assert pattern == 'follow_up_queries'
    
    def test_detect_query_patterns_clarification(self, context_planner):
        """Test detection of clarification request patterns"""
        pattern = context_planner._detect_query_patterns("Can you explain what you mean by complications?")
        assert pattern == 'clarification_requests'
    
    def test_detect_query_patterns_preference(self, context_planner):
        """Test detection of preference indicator patterns"""
        pattern = context_planner._detect_query_patterns("I'm allergic to penicillin")
        assert pattern == 'preference_indicators'
    
    def test_detect_query_patterns_independent(self, context_planner):
        """Test detection of independent queries"""
        pattern = context_planner._detect_query_patterns("What is hypertension?")
        assert pattern == 'independent_query'
    
    def test_extract_entities_from_text(self, context_planner):
        """Test basic entity extraction from text"""
        entities = context_planner._extract_entities_from_text("Patient has diabetes and takes insulin for pain")
        
        assert 'diabetes' in entities
        assert 'insulin' in entities
        assert 'pain' in entities
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_relevance_with_overlap(self, context_planner, sample_query, sample_context):
        """Test conversation relevance analysis with entity overlap"""
        conversation_history = sample_context['conversation_history']
        
        result = await context_planner._analyze_conversation_relevance(sample_query, conversation_history)
        
        assert result['conversation_relevance'] > 0.0
        assert len(result['relevant_history']) > 0
        assert result['entity_continuity'] > 0.0
        assert 'diabetes' in result['context_entities']
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_relevance_empty_history(self, context_planner, sample_query):
        """Test conversation relevance analysis with empty history"""
        result = await context_planner._analyze_conversation_relevance(sample_query, [])
        
        assert result['conversation_relevance'] == 0.0
        assert result['relevant_history'] == []
        assert result['entity_continuity'] == 0.0
    
    def test_analyze_preference_relevance_with_matches(self, context_planner, sample_context):
        """Test preference relevance analysis with matching preferences"""
        query = MedicalQuery(
            query_text="What medications are safe for someone allergic to penicillin?",
            query_type=QueryType.DRUG_INFO
        )
        
        result = context_planner._analyze_preference_relevance(query, sample_context['user_preferences'])
        
        assert result['preference_relevance'] > 0.0
        assert 'allergies' in result['relevant_preferences']
        assert 'penicillin' in result['relevant_preferences']['allergies']
    
    def test_analyze_preference_relevance_no_matches(self, context_planner, sample_query, sample_context):
        """Test preference relevance analysis with no matching preferences"""
        result = context_planner._analyze_preference_relevance(sample_query, sample_context['user_preferences'])
        
        # Should still include language and explanation_level as they're always relevant
        assert result['preference_relevance'] > 0.0
        assert 'language' in result['relevant_preferences']
        assert 'explanation_level' in result['relevant_preferences']
    
    def test_determine_integration_strategies_follow_up(self, context_planner):
        """Test integration strategy determination for follow-up queries"""
        context_analysis = {
            'query_pattern_match': 'follow_up_queries',
            'preference_relevance': 0.2,
            'conversation_relevance': 0.4
        }
        
        strategies = context_planner._determine_integration_strategies(context_analysis)
        
        assert 'entity_enrichment' in strategies
        assert 'conversation_threading' in strategies
    
    def test_determine_integration_strategies_preferences(self, context_planner):
        """Test integration strategy determination for preference-heavy queries"""
        context_analysis = {
            'query_pattern_match': 'preference_indicators',
            'preference_relevance': 0.8,
            'conversation_relevance': 0.2
        }
        
        strategies = context_planner._determine_integration_strategies(context_analysis)
        
        assert 'preference_filtering' in strategies
    
    def test_apply_entity_enrichment(self, context_planner, sample_base_plan):
        """Test entity enrichment application"""
        context_analysis = {
            'context_entities': ['diabetes', 'insulin', 'blood sugar']
        }
        
        enhanced_plan = context_planner._apply_entity_enrichment(sample_base_plan, context_analysis)
        
        # Should add new entities to key_entities
        assert 'insulin' in enhanced_plan.key_entities
        assert 'blood sugar' in enhanced_plan.key_entities
        
        # Should add context entities to execution steps
        for step in enhanced_plan.execution_steps:
            assert 'context_entities' in step['input_data']
            assert step['context_enriched'] is True
    
    def test_apply_preference_filtering(self, context_planner, sample_base_plan):
        """Test preference filtering application"""
        context_analysis = {
            'relevant_preferences': {
                'allergies': ['penicillin'],
                'language': 'english'
            }
        }
        
        enhanced_plan = context_planner._apply_preference_filtering(sample_base_plan, context_analysis)
        
        # Should add preferences to execution steps
        for step in enhanced_plan.execution_steps:
            assert 'user_preferences' in step['input_data']
            assert step['preference_filtered'] is True
    
    def test_apply_preference_filtering_drug_query(self, context_planner):
        """Test preference filtering for drug-related queries with allergies"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.DRUG_INFO,
            tools_required=['get_drug_information'],
            execution_steps=[
                {
                    'step_number': 1,
                    'tool_name': 'get_drug_information',
                    'estimated_time': 3.0
                }
            ]
        )
        
        context_analysis = {
            'relevant_preferences': {
                'allergies': ['penicillin', 'sulfa']
            }
        }
        
        enhanced_plan = context_planner._apply_preference_filtering(plan, context_analysis)
        
        # Should add allergy check for drug information queries
        drug_step = enhanced_plan.execution_steps[0]
        assert 'allergy_check' in drug_step['input_data']
        assert drug_step['safety_enhanced'] is True
    
    def test_apply_conversation_threading(self, context_planner, sample_base_plan):
        """Test conversation threading application"""
        context_analysis = {
            'relevant_history': [
                {'interaction': {'query_text': 'Previous query'}, 'relevance_score': 0.8},
                {'interaction': {'query_text': 'Another query'}, 'relevance_score': 0.6}
            ]
        }
        
        enhanced_plan = context_planner._apply_conversation_threading(sample_base_plan, context_analysis)
        
        # Should add conversation context to execution steps
        for step in enhanced_plan.execution_steps:
            assert 'conversation_context' in step['input_data']
            assert step['conversation_threaded'] is True
        
        # Should update context requirements
        assert enhanced_plan.context_requirements['conversation_threading'] is True
        assert enhanced_plan.context_requirements['thread_length'] > 0
    
    def test_apply_context_disambiguation(self, context_planner, sample_base_plan):
        """Test context disambiguation application"""
        context_analysis = {
            'conversation_relevance': 0.7,  # Above threshold
            'relevant_history': [{'interaction': {'query_text': 'Previous'}}],
            'context_entities': ['diabetes']
        }
        
        enhanced_plan = context_planner._apply_context_disambiguation(sample_base_plan, context_analysis)
        
        # Should add disambiguation context to execution steps
        for step in enhanced_plan.execution_steps:
            assert 'disambiguation_context' in step['input_data']
            assert step['context_disambiguated'] is True
        
        # Should boost confidence
        assert enhanced_plan.confidence > sample_base_plan.confidence
    
    @pytest.mark.asyncio
    async def test_optimize_for_context_continuity_high_continuity(self, context_planner):
        """Test context continuity optimization with high continuity score"""
        plan = QueryPlan(
            query_id="test-123",
            query_type=QueryType.SYMPTOMS,
            tools_required=['search_medical_term', 'analyze_symptoms'],
            execution_steps=[
                {'step_number': 1, 'tool_name': 'search_medical_term'},
                {'step_number': 2, 'tool_name': 'analyze_symptoms'}
            ]
        )
        
        context_analysis = {
            'entity_continuity': 0.8  # High continuity
        }
        
        optimized_plan = await context_planner._optimize_for_context_continuity(plan, context_analysis)
        
        # Should prioritize context-aware tools (analyze_symptoms should come first)
        assert optimized_plan.tools_required[0] == 'analyze_symptoms'
        assert optimized_plan.execution_steps[0]['tool_name'] == 'analyze_symptoms'
        
        # Should add continuity metadata
        assert optimized_plan.context_requirements['context_continuity_score'] == 0.8
        assert optimized_plan.context_requirements['context_optimized'] is True
    
    @pytest.mark.asyncio
    async def test_enhance_plan_with_context(self, context_planner, sample_base_plan, sample_query, sample_context):
        """Test plan enhancement with context"""
        context_analysis = {
            'integration_strategies': ['entity_enrichment', 'preference_filtering'],
            'conversation_relevance': 0.6,
            'preference_relevance': 0.4,
            'entity_continuity': 0.5,
            'context_entities': ['diabetes', 'insulin'],
            'relevant_preferences': {'language': 'english'}
        }
        
        enhanced_plan = await context_planner._enhance_plan_with_context(
            sample_base_plan, sample_query, sample_context, context_analysis
        )
        
        # Should boost confidence based on context
        assert enhanced_plan.confidence > sample_base_plan.confidence
        
        # Should update reasoning with integration info
        assert 'Context integration' in enhanced_plan.reasoning
        
        # Should have enhanced entities
        assert 'insulin' in enhanced_plan.key_entities
    
    @pytest.mark.asyncio
    async def test_analyze_context_relevance_full(self, context_planner, sample_query, sample_context):
        """Test complete context relevance analysis"""
        analysis = await context_planner._analyze_context_relevance(sample_query, sample_context)
        
        assert 'conversation_relevance' in analysis
        assert 'preference_relevance' in analysis
        assert 'entity_continuity' in analysis
        assert 'query_pattern_match' in analysis
        assert 'integration_strategies' in analysis
        
        # Should detect follow-up pattern
        assert analysis['query_pattern_match'] == 'follow_up_queries'
        
        # Should have some conversation relevance due to diabetes entity overlap
        assert analysis['conversation_relevance'] > 0.0
    
    @pytest.mark.asyncio
    async def test_create_context_aware_plan_success(self, context_planner, sample_query, sample_context, sample_base_plan):
        """Test successful context-aware plan creation"""
        # Mock the base planner to return our sample plan
        context_planner.query_planner.analyze_query.return_value = sample_base_plan
        
        plan = await context_planner.create_context_aware_plan(sample_query, sample_context)
        
        assert isinstance(plan, QueryPlan)
        assert plan.query_id == sample_query.id
        
        # Should have called the base planner
        context_planner.query_planner.analyze_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_context_aware_plan_no_context(self, context_planner, sample_query, sample_base_plan):
        """Test context-aware plan creation without context"""
        context_planner.query_planner.analyze_query.return_value = sample_base_plan
        
        plan = await context_planner.create_context_aware_plan(sample_query, None)
        
        assert isinstance(plan, QueryPlan)
        # Should still work without context
        context_planner.query_planner.analyze_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_context_aware_plan_exception_handling(self, context_planner, sample_query, sample_base_plan):
        """Test context-aware plan creation with exception handling"""
        # Mock the base planner to raise exception initially, then return plan
        context_planner.query_planner.analyze_query.side_effect = [Exception("Analysis failed"), sample_base_plan]
        
        plan = await context_planner.create_context_aware_plan(sample_query, {})
        
        # Should fallback to basic planning
        assert isinstance(plan, QueryPlan)
        assert context_planner.query_planner.analyze_query.call_count == 2
    
    def test_get_context_statistics(self, context_planner):
        """Test getting context statistics"""
        stats = context_planner.get_context_statistics()
        
        assert 'context_patterns' in stats
        assert 'integration_strategies' in stats
        assert 'relevance_weights' in stats
        assert 'pattern_types' in stats
        assert 'strategy_types' in stats
        assert 'memory_manager_available' in stats
        
        assert stats['context_patterns'] > 0
        assert stats['integration_strategies'] > 0
        assert stats['memory_manager_available'] is True
        assert len(stats['pattern_types']) > 0
        assert len(stats['strategy_types']) > 0