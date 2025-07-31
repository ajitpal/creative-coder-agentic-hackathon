"""
Context-aware planner for Intelligent Healthcare Navigator
Integrates conversation memory and user preferences into planning
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re # Added for improved entity extraction

from src.planner import QueryPlanner, QueryPlan, UrgencyLevel
from src.models import MedicalQuery, QueryType
from src.utils import setup_logging

logger = setup_logging()

class ContextAwarePlanner:
    """Context-aware planning that integrates conversation memory and user preferences"""
    
    def __init__(self, query_planner: QueryPlanner, memory_manager=None):
        """Initialize context-aware planner"""
        self.query_planner = query_planner
        self.memory_manager = memory_manager
        
        # Context analysis patterns
        self.context_patterns = self._load_context_patterns()
        
        # Relevance scoring weights
        self.relevance_weights = self._load_relevance_weights()
        
        # Integration strategies
        self.integration_strategies = self._load_integration_strategies()
        
        logger.info("Context-aware planner initialized")
    
    def _load_context_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for context analysis"""
        return {
            'follow_up_queries': {
                'patterns': [
                    r'tell me more about',
                    r'what about',
                    r'and what',
                    r'also',
                    r'additionally',
                    r'furthermore'
                ],
                'confidence_boost': 0.3,
                'requires_history': True
            },
            'clarification_requests': {
                'patterns': [
                    r'what do you mean',
                    r'can you explain',
                    r'i don\'t understand',
                    r'clarify',
                    r'elaborate'
                ],
                'confidence_boost': 0.4,
                'requires_history': True
            },
            'related_queries': {
                'patterns': [
                    r'similar to',
                    r'like the previous',
                    r'same as',
                    r'related to'
                ],
                'confidence_boost': 0.2,
                'requires_history': True
            },
            'preference_indicators': {
                'patterns': [
                    r'i prefer',
                    r'i like',
                    r'i usually',
                    r'my doctor says',
                    r'i\'m allergic to'
                ],
                'confidence_boost': 0.1,
                'requires_preferences': True
            }
        }
    
    def _load_relevance_weights(self) -> Dict[str, float]:
        """Load weights for relevance scoring"""
        return {
            'temporal_decay': 0.1,  # How much relevance decreases over time
            'semantic_similarity': 0.4,  # Weight for semantic similarity
            'entity_overlap': 0.3,  # Weight for entity overlap
            'query_type_match': 0.2,  # Weight for matching query types
            'user_preference_match': 0.3,  # Weight for user preference alignment
            'conversation_continuity': 0.5  # Weight for conversation flow
        }
    
    def _load_integration_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load context integration strategies"""
        return {
            'entity_enrichment': {
                'description': 'Enrich current query with entities from context',
                'applicable_when': ['follow_up_queries', 'related_queries'],
                'max_entities': 5
            },
            'preference_filtering': {
                'description': 'Filter results based on user preferences',
                'applicable_when': ['preference_indicators'],
                'preference_types': ['allergies', 'language', 'medication_preferences']
            },
            'conversation_threading': {
                'description': 'Thread conversation for coherent responses',
                'applicable_when': ['follow_up_queries', 'clarification_requests'],
                'max_history_items': 3
            },
            'context_disambiguation': {
                'description': 'Use context to disambiguate unclear queries',
                'applicable_when': ['ambiguous_queries'],
                'disambiguation_threshold': 0.6
            }
        }
    
    async def create_context_aware_plan(
        self, 
        query: MedicalQuery, 
        context: Dict[str, Any]
    ) -> QueryPlan:
        """Create context-aware execution plan"""
        logger.info(f"Creating context-aware plan for query {query.id}")
        
        try:
            # Step 1: Analyze context relevance
            context_analysis = await self._analyze_context_relevance(query, context)
            
            # Step 2: Create base plan using query planner
            base_plan = await self.query_planner.analyze_query(query, context)
            
            # Step 3: Enhance plan with context
            enhanced_plan = self._enhance_plan_with_context(base_plan, context_analysis)
            
            # Step 4: Optimize for context continuity
            final_plan = await self._optimize_for_context_continuity(enhanced_plan, context_analysis)
            
            logger.info(f"Context-aware plan created with {len(final_plan.execution_steps)} steps")
            return final_plan
            
        except Exception as e:
            logger.error(f"Context-aware planning failed: {e}")
            # Fallback to basic planning
            return await self.query_planner.analyze_query(query, context)
    
    async def _analyze_context_relevance(
        self, 
        query: MedicalQuery, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze relevance of available context"""
        analysis = {
            'conversation_relevance': 0.0,
            'preference_relevance': 0.0,
            'entity_continuity': 0.0,
            'query_pattern_match': '',
            'relevant_history': [],
            'relevant_preferences': {},
            'context_entities': [],
            'integration_strategies': []
        }
        
        # Analyze conversation history
        if context and 'conversation_history' in context:
            conversation_history = context['conversation_history']
            if conversation_history:
                analysis.update(await self._analyze_conversation_relevance(query, conversation_history))
        
        # Analyze user preferences
        if context and 'user_preferences' in context:
            user_preferences = context['user_preferences']
            if user_preferences:
                analysis.update(self._analyze_preference_relevance(query, user_preferences))
        
        # Detect query patterns
        analysis['query_pattern_match'] = self._detect_query_patterns(query.query_text)
        
        # Determine integration strategies
        analysis['integration_strategies'] = self._determine_integration_strategies(analysis)
        
        return analysis
    
    async def _analyze_conversation_relevance(
        self, 
        query: MedicalQuery, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relevance of conversation history"""
        if not conversation_history:
            return {
                'conversation_relevance': 0.0,
                'relevant_history': [],
                'entity_continuity': 0.0
            }
        
        # Extract entities from current query
        current_entities = set(self._extract_entities_from_text(query.query_text))
        
        relevant_items = []
        all_shared_entities = set()
        
        # Analyze last 10 interactions
        for i, interaction in enumerate(conversation_history[-10:]):
            # Temporal decay (more recent = more relevant)
            time_weight = 1.0 - (i * self.relevance_weights['temporal_decay'])
            
            # Extract entities from historical interaction
            hist_query_text = interaction.get('query', '') + ' ' + interaction.get('response_text', '')
            hist_entities = set(self._extract_entities_from_text(hist_query_text))
            
            # Calculate entity overlap
            shared_entities = current_entities.intersection(hist_entities)
            entity_overlap = len(shared_entities) / max(len(current_entities), 1)
            
            # Query type match
            hist_query_type = interaction.get('query_type', '')
            type_match = 1.0 if hist_query_type == query.query_type.value else 0.0
            
            # Calculate total relevance score
            relevance_score = (
                time_weight * 0.3 +
                entity_overlap * self.relevance_weights['entity_overlap'] +
                type_match * self.relevance_weights['query_type_match']
            )
            
            if relevance_score > 0.3:  # Relevance threshold
                relevant_items.append({
                    'interaction': interaction,
                    'relevance_score': relevance_score,
                    'shared_entities': list(shared_entities),
                    'entity_overlap': entity_overlap
                })
                all_shared_entities.update(shared_entities)
        
        # Sort by relevance
        relevant_items.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Calculate overall conversation relevance
        avg_relevance = sum(item['relevance_score'] for item in relevant_items) / len(relevant_items) if relevant_items else 0.0
        
        # Calculate entity continuity
        entity_continuity = len(all_shared_entities) / max(len(current_entities), 1)
        
        return {
            'conversation_relevance': avg_relevance,
            'relevant_history': relevant_items[:5],  # Top 5 most relevant
            'entity_continuity': entity_continuity,
            'context_entities': list(all_shared_entities)
        }
    
    def _analyze_preference_relevance(
        self, 
        query: MedicalQuery, 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze relevance of user preferences"""
        if not user_preferences:
            return {
                'preference_relevance': 0.0,
                'relevant_preferences': {}
            }
        
        query_lower = query.query_text.lower()
        relevant_prefs = {}
        
        # Check for allergy-related preferences
        if 'allergies' in user_preferences:
            allergies = user_preferences['allergies']
            if any(allergy.lower() in query_lower for allergy in allergies):
                relevant_prefs['allergies'] = allergies
        
        # Check for medication preferences
        if 'medication_preferences' in user_preferences:
            if 'drug' in query_lower or 'medication' in query_lower:
                relevant_prefs['medication_preferences'] = user_preferences['medication_preferences']
        
        # Check for language preferences
        if 'language' in user_preferences:
            relevant_prefs['language'] = user_preferences['language']
        
        # Check for complexity preferences
        if 'explanation_level' in user_preferences:
            relevant_prefs['explanation_level'] = user_preferences['explanation_level']
        
        # Calculate overall preference relevance
        preference_relevance = len(relevant_prefs) / max(len(user_preferences), 1)
        
        return {
            'preference_relevance': preference_relevance,
            'relevant_preferences': relevant_prefs
        }
    
    def _detect_query_patterns(self, query_text: str) -> str:
        """Detect query patterns in text"""
        query_lower = query_text.lower()
        
        for pattern_name, pattern_info in self.context_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in query_lower:
                    return pattern_name
        
        return 'independent_query'
    
    def _determine_integration_strategies(self, analysis: Dict[str, Any]) -> List[str]:
        """Determine which integration strategies to apply"""
        strategies = []
        
        for strategy_name, strategy_info in self.integration_strategies.items():
            applicable_patterns = strategy_info.get('applicable_when', [])
            
            # Check if current query pattern matches
            if analysis.get('query_pattern_match') in applicable_patterns:
                strategies.append(strategy_name)
            
            # Special cases
            if strategy_name == 'context_disambiguation' and analysis.get('conversation_relevance', 0) > 0.5:
                strategies.append(strategy_name)
            
            if strategy_name == 'preference_filtering' and analysis.get('preference_relevance', 0) > 0.3:
                strategies.append(strategy_name)
        
        return strategies
    
    def _enhance_plan_with_context(
        self, 
        base_plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Enhance base plan with context information"""
        enhanced_plan = QueryPlan(
            query_id=base_plan.query_id,
            query_type=base_plan.query_type,
            tools_required=base_plan.tools_required.copy(),
            execution_steps=base_plan.execution_steps.copy(),
            priority=base_plan.priority,
            urgency_level=base_plan.urgency_level,
            confidence=base_plan.confidence,
            reasoning=base_plan.reasoning,
            key_entities=base_plan.key_entities.copy(),
            context_requirements=base_plan.context_requirements.copy(),
            estimated_complexity=base_plan.estimated_complexity,
            medical_disclaimers=base_plan.medical_disclaimers.copy()
        )
        
        # Apply integration strategies
        for strategy in context_analysis.get('integration_strategies', []):
            if strategy == 'entity_enrichment':
                enhanced_plan = self._apply_entity_enrichment(enhanced_plan, context_analysis)
            elif strategy == 'preference_filtering':
                enhanced_plan = self._apply_preference_filtering(enhanced_plan, context_analysis)
            elif strategy == 'conversation_threading':
                enhanced_plan = self._apply_conversation_threading(enhanced_plan, context_analysis)
            elif strategy == 'context_disambiguation':
                enhanced_plan = self._apply_context_disambiguation(enhanced_plan, context_analysis)
        
        # Update confidence based on context
        context_confidence_boost = (
            context_analysis.get('conversation_relevance', 0) * 0.1 +
            context_analysis.get('preference_relevance', 0) * 0.05 +
            context_analysis.get('entity_continuity', 0) * 0.1
        )
        enhanced_plan.confidence = min(enhanced_plan.confidence + context_confidence_boost, 1.0)
        
        # Update reasoning
        enhanced_plan.reasoning += f" [Context integration: {', '.join(context_analysis.get('integration_strategies', []))}]"
        
        return enhanced_plan
    
    def _apply_entity_enrichment(
        self, 
        plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Apply entity enrichment from context"""
        context_entities = context_analysis.get('context_entities', [])
        max_entities = self.integration_strategies['entity_enrichment']['max_entities']
        
        # Add relevant context entities
        for entity in context_entities[:max_entities]:
            if entity not in plan.key_entities:
                plan.key_entities.append(entity)
        
        # Update execution steps with enriched entities
        for step in plan.execution_steps:
            if 'input_data' not in step:
                step['input_data'] = {}
            step['input_data']['context_entities'] = context_entities[:max_entities]
            step['context_enhanced'] = True
        
        return plan
    
    def _apply_preference_filtering(
        self, 
        plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Apply preference-based filtering"""
        relevant_prefs = context_analysis.get('relevant_preferences', {})
        
        # Add preference information to execution steps
        for step in plan.execution_steps:
            if 'input_data' not in step:
                step['input_data'] = {}
            step['input_data']['user_preferences'] = relevant_prefs
            step['preference_filtered'] = True
            
            # Special handling for drug-related queries with allergies
            if step.get('tool_name') == 'get_drug_information' and 'allergies' in relevant_prefs:
                step['input_data']['allergy_check'] = relevant_prefs['allergies']
                step['safety_enhanced'] = True
        
        return plan
    
    def _apply_conversation_threading(
        self, 
        plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Apply conversation threading"""
        relevant_history = context_analysis.get('relevant_history', [])
        max_history = self.integration_strategies['conversation_threading']['max_history_items']
        
        # Add conversation context to execution steps
        for step in plan.execution_steps:
            if 'input_data' not in step:
                step['input_data'] = {}
            step['input_data']['conversation_context'] = relevant_history[:max_history]
            step['conversation_threaded'] = True
        
        return plan
    
    def _apply_context_disambiguation(
        self, 
        plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Apply context-based disambiguation"""
        disambiguation_threshold = self.integration_strategies['context_disambiguation']['disambiguation_threshold']
        
        if context_analysis.get('conversation_relevance', 0) >= disambiguation_threshold:
            # Add disambiguation context to steps
            for step in plan.execution_steps:
                if 'input_data' not in step:
                    step['input_data'] = {}
                step['input_data']['disambiguation_context'] = {
                    'relevant_history': context_analysis.get('relevant_history', [])[:2],
                    'shared_entities': context_analysis.get('context_entities', [])
                }
                step['disambiguated'] = True
        
        return plan
    
    async def _optimize_for_context_continuity(
        self, 
        plan: QueryPlan, 
        context_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Optimize plan for context continuity"""
        continuity_score = context_analysis.get('entity_continuity', 0)
        
        if continuity_score > 0.5:  # High continuity
            # Prioritize tools that can leverage context
            context_aware_tools = ['analyze_symptoms', 'search_medical_term']
            
            # Reorder tools to prioritize context-aware ones
            reordered_tools = []
            other_tools = []
            
            for tool in plan.tools_required:
                if tool in context_aware_tools:
                    reordered_tools.append(tool)
                else:
                    other_tools.append(tool)
            
            plan.tools_required = reordered_tools + other_tools
            
            # Update execution steps order
            new_steps = []
            step_number = 1
            
            for tool in plan.tools_required:
                for step in plan.execution_steps:
                    if step.get('tool_name') == tool:
                        step['step_number'] = step_number
                        new_steps.append(step)
                        step_number += 1
                        break
            
            plan.execution_steps = new_steps
            
            # Add continuity metadata
            plan.context_requirements['context_optimized'] = True
            plan.context_requirements['continuity_score'] = continuity_score
        
        return plan
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract basic entities from text using generic NLP approaches"""
        entities = []
        
        text_lower = text.lower()
        
        # Generic symptom extraction patterns - more flexible
        symptom_patterns = [
            # "I have X" patterns
            r'i have (\w+(?:\s+\w+)*)',
            r'i\'m having (\w+(?:\s+\w+)*)',
            r'i am having (\w+(?:\s+\w+)*)',
            r'i\'ve been having (\w+(?:\s+\w+)*)',
            r'i have been having (\w+(?:\s+\w+)*)',
            
            # "symptoms:" patterns
            r'symptoms?[:\s]+(\w+(?:\s+\w+)*)',
            r'my symptoms?[:\s]+(\w+(?:\s+\w+)*)',
            
            # "experiencing" patterns
            r'i\'m experiencing (\w+(?:\s+\w+)*)',
            r'i am experiencing (\w+(?:\s+\w+)*)',
            r'experiencing (\w+(?:\s+\w+)*)',
            
            # "feeling" patterns
            r'i\'m feeling (\w+(?:\s+\w+)*)',
            r'i am feeling (\w+(?:\s+\w+)*)',
            r'feeling (\w+(?:\s+\w+)*)',
            
            # "suffering from" patterns
            r'suffering from (\w+(?:\s+\w+)*)',
            r'i\'m suffering from (\w+(?:\s+\w+)*)',
            
            # Generic pain patterns
            r'(\w+(?:\s+\w+)*)\s+pain',
            r'pain in (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+ache',
            r'ache in (\w+(?:\s+\w+)*)',
            
            # Generic symptom patterns
            r'(\w+(?:\s+\w+)*)\s+discomfort',
            r'(\w+(?:\s+\w+)*)\s+problem',
            r'(\w+(?:\s+\w+)*)\s+issue',
            r'(\w+(?:\s+\w+)*)\s+trouble',
            
            # Drug/medication patterns
            r'tell me about (\w+(?:\s+\w+)*)',
            r'what about (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+side effects',
            r'(\w+(?:\s+\w+)*)\s+safety',
            r'(\w+(?:\s+\w+)*)\s+recall',
            r'drug (\w+(?:\s+\w+)*)',
            r'medication (\w+(?:\s+\w+)*)',
            r'medicine (\w+(?:\s+\w+)*)',
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match and len(match) > 2:
                    # Clean up the match and add it
                    clean_match = match.strip()
                    if clean_match and len(clean_match) > 2:
                        entities.append(clean_match)
        
        # Generic symptom phrase extraction after common indicators
        symptom_indicators = [
            'symptoms:', 'having', 'experiencing', 'feeling', 'suffering',
            'problem with', 'issue with', 'trouble with', 'discomfort in',
            'pain in', 'ache in', 'problem in', 'issue in'
        ]
        
        for indicator in symptom_indicators:
            if indicator in text_lower:
                # Find text after the indicator
                parts = text_lower.split(indicator)
                if len(parts) > 1:
                    symptom_text = parts[1].strip()
                    # Extract meaningful symptom phrases (2-4 words)
                    words = symptom_text.split()
                    if len(words) >= 2:
                        # Look for meaningful symptom phrases
                        for i in range(len(words) - 1):
                            for phrase_length in [2, 3, 4]:  # Try 2, 3, or 4 word phrases
                                if i + phrase_length <= len(words):
                                    phrase = ' '.join(words[i:i+phrase_length])
                                    # Filter out common stop words and short phrases
                                    if (len(phrase) > 3 and 
                                        not any(word in ['and', 'or', 'with', 'the', 'a', 'an', 'my', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had'] 
                                               for word in words[i:i+phrase_length])):
                                        entities.append(phrase)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted_terms)
        
        # Extract capitalized terms (potential proper nouns) - but be more selective
        # Only extract if they're not common words
        common_words = {
            'tell', 'what', 'how', 'when', 'where', 'why', 'about', 'side', 'effects', 
            'safety', 'recall', 'i', 'am', 'have', 'having', 'these', 'symptoms',
            'my', 'is', 'are', 'was', 'were', 'been', 'has', 'had', 'the', 'a', 'an',
            'and', 'or', 'with', 'in', 'on', 'at', 'to', 'for', 'of', 'from'
        }
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', text)
        for term in capitalized_terms:
            if term.lower() not in common_words and len(term) > 2:
                entities.append(term)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about context-aware planning capabilities"""
        return {
            'context_patterns': len(self.context_patterns),
            'relevance_weights': len(self.relevance_weights),
            'integration_strategies': len(self.integration_strategies),
            'memory_manager_available': self.memory_manager is not None,
            'pattern_types': list(self.context_patterns.keys()),
            'strategy_types': list(self.integration_strategies.keys())
        }