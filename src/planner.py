"""
Planner module for Intelligent Healthcare Navigator
Implements the Reasoning phase of the ReAct pattern using Gemini API
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

from src.models import QueryType, MedicalQuery
from src.utils import setup_logging, sanitize_input

logger = setup_logging()

class UrgencyLevel(Enum):
    """Query urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"

class ConfidenceLevel(Enum):
    """Analysis confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class QueryPlan:
    """Execution plan for a medical query"""
    query_id: str
    query_type: QueryType
    tools_required: List[str] = field(default_factory=list)
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 1
    urgency_level: UrgencyLevel = UrgencyLevel.LOW
    confidence: float = 0.0
    reasoning: str = ""
    key_entities: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: str = "simple"
    medical_disclaimers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary"""
        return {
            'query_id': self.query_id,
            'query_type': self.query_type.value,
            'tools_required': self.tools_required,
            'execution_steps': self.execution_steps,
            'priority': self.priority,
            'urgency_level': self.urgency_level.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'key_entities': self.key_entities,
            'context_requirements': self.context_requirements,
            'estimated_complexity': self.estimated_complexity,
            'medical_disclaimers': self.medical_disclaimers
        }

class QueryPlanner:
    """Analyzes queries and creates execution plans using Gemini API"""
    
    def __init__(self, gemini_client=None):
        """Initialize query planner"""
        self.gemini_client = gemini_client
        
        # Query classification patterns
        self.classification_patterns = self._load_classification_patterns()
        
        # Tool mapping
        self.available_tools = self._load_available_tools()
        
        # Emergency keywords
        self.emergency_keywords = self._load_emergency_keywords()
        
        # Medical disclaimer templates
        self.disclaimer_templates = self._load_disclaimer_templates()
        
        logger.info("Query planner initialized")
    
    def _load_classification_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for query classification"""
        return {
            'medical_term': {
                'keywords': [
                    'what is', 'define', 'definition', 'meaning', 'explain',
                    'disease', 'condition', 'syndrome', 'disorder', 'illness'
                ],
                'patterns': [
                    r'what is (\w+)',
                    r'define (\w+)',
                    r'explain (\w+)',
                    r'tell me about (\w+)',
                    r'(\w+) definition'
                ],
                'confidence_boost': 0.2
            },
            'drug_info': {
                'keywords': [
                    'drug', 'medication', 'medicine', 'pill', 'tablet',
                    'recall', 'side effects', 'adverse', 'safety', 'dosage'
                ],
                'patterns': [
                    r'(\w+) recall',
                    r'(\w+) side effects',
                    r'(\w+) safety',
                    r'drug (\w+)',
                    r'medication (\w+)'
                ],
                'confidence_boost': 0.3
            },
            'symptoms': {
                'keywords': [
                    'symptoms', 'feeling', 'experiencing', 'pain', 'ache',
                    'headache', 'fever', 'nausea', 'dizzy', 'tired', 'fatigue'
                ],
                'patterns': [
                    r'i have (\w+)',
                    r'experiencing (\w+)',
                    r'feeling (\w+)',
                    r'symptoms of (\w+)',
                    r'(\w+) pain'
                ],
                'confidence_boost': 0.25
            },
            'document_summary': {
                'keywords': [
                    'summarize', 'summary', 'document', 'report', 'analyze',
                    'prescription', 'lab results', 'medical record'
                ],
                'patterns': [
                    r'summarize (\w+)',
                    r'analyze (\w+)',
                    r'what does this (\w+) say',
                    r'explain this (\w+)'
                ],
                'confidence_boost': 0.3
            },
            'entity_extraction': {
                'keywords': [
                    'extract', 'find', 'identify', 'list', 'entities',
                    'medications in', 'diseases in', 'conditions in'
                ],
                'patterns': [
                    r'extract (\w+) from',
                    r'find (\w+) in',
                    r'list (\w+) mentioned',
                    r'identify (\w+)'
                ],
                'confidence_boost': 0.2
            }
        }
    
    def _load_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load available tools and their capabilities"""
        return {
            'search_medical_term': {
                'description': 'Search for medical term definitions using WHO ICD API',
                'input_types': ['medical_term'],
                'complexity': 'simple',
                'estimated_time': 2.0,
                'requires_context': False
            },
            'get_drug_information': {
                'description': 'Get drug safety information from OpenFDA API',
                'input_types': ['drug_info'],
                'complexity': 'medium',
                'estimated_time': 3.0,
                'requires_context': False
            },
            'analyze_symptoms': {
                'description': 'Analyze symptoms using Gemini AI (no diagnosis)',
                'input_types': ['symptoms'],
                'complexity': 'medium',
                'estimated_time': 2.5,
                'requires_context': True
            },
            'extract_medical_entities': {
                'description': 'Extract medical entities from text using NLP',
                'input_types': ['entity_extraction'],
                'complexity': 'simple',
                'estimated_time': 1.5,
                'requires_context': False
            },
            'summarize_medical_document': {
                'description': 'Summarize medical documents',
                'input_types': ['document_summary'],
                'complexity': 'complex',
                'estimated_time': 4.0,
                'requires_context': False
            }
        }
    
    def _load_emergency_keywords(self) -> List[str]:
        """Load keywords that indicate emergency situations"""
        return [
            'emergency', 'urgent', 'critical', 'severe', 'acute',
            'chest pain', 'heart attack', 'stroke', 'difficulty breathing',
            'unconscious', 'bleeding', 'overdose', 'poisoning',
            'severe pain', 'can\'t breathe', 'choking', 'seizure'
        ]
    
    def _load_disclaimer_templates(self) -> Dict[str, str]:
        """Load medical disclaimer templates"""
        return {
            'general': "This information is for educational purposes only and should not replace professional medical advice.",
            'symptoms': "This analysis does not constitute medical diagnosis. Consult healthcare professionals for medical concerns.",
            'emergency': "If this is a medical emergency, contact emergency services immediately (911 in US).",
            'drug_safety': "Always consult healthcare providers before making medication decisions.",
            'diagnosis': "This system does not provide medical diagnoses. Seek professional medical evaluation."
        }
    
    async def analyze_query(self, query: MedicalQuery, context: Dict[str, Any] = None) -> QueryPlan:
        """Analyze query and create execution plan"""
        logger.info(f"Analyzing query: {query.query_text[:50]}...")
        
        try:
            # Step 1: Basic classification using patterns
            basic_classification = self._classify_query_basic(query.query_text)
            
            # Step 2: Enhanced analysis using Gemini if available
            if self.gemini_client:
                gemini_analysis = await self._analyze_with_gemini(query, context, basic_classification)
            else:
                gemini_analysis = basic_classification
            
            # Step 3: Determine urgency level
            urgency_level = self._assess_urgency(query.query_text, gemini_analysis)
            
            # Step 4: Select appropriate tools
            tools_required = self._determine_tools_needed(gemini_analysis['query_type'], gemini_analysis)
            
            # Step 5: Create execution steps
            execution_steps = self._create_execution_steps(tools_required, gemini_analysis, context)
            
            # Step 6: Add medical disclaimers
            disclaimers = self._select_disclaimers(gemini_analysis['query_type'], urgency_level)
            
            # Step 7: Calculate priority
            priority = self._calculate_priority(urgency_level, gemini_analysis['confidence'])
            
            # Create the plan
            plan = QueryPlan(
                query_id=query.id,
                query_type=QueryType(gemini_analysis['query_type']),
                tools_required=tools_required,
                execution_steps=execution_steps,
                priority=priority,
                urgency_level=urgency_level,
                confidence=gemini_analysis['confidence'],
                reasoning=gemini_analysis.get('reasoning', ''),
                key_entities=gemini_analysis.get('key_entities', []),
                context_requirements=self._determine_context_requirements(tools_required),
                estimated_complexity=self._estimate_complexity(tools_required),
                medical_disclaimers=disclaimers
            )
            
            logger.info(f"Query analysis complete: {plan.query_type.value} (confidence: {plan.confidence:.2f})")
            return plan
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return fallback plan
            return self._create_fallback_plan(query)
    
    def _classify_query_basic(self, query_text: str) -> Dict[str, Any]:
        """Basic query classification using patterns"""
        query_lower = query_text.lower()
        scores = {}
        
        # Score each query type
        for query_type, patterns in self.classification_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1
            
            # Check regex patterns
            for pattern in patterns['patterns']:
                if re.search(pattern, query_lower):
                    score += 2
            
            # Apply confidence boost
            if score > 0:
                score += patterns['confidence_boost']
            
            scores[query_type] = score
        
        # Determine best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type] / 5.0, 1.0)  # Normalize to 0-1
        else:
            best_type = 'medical_term'  # Default
            confidence = 0.3
        
        # Extract potential entities
        entities = self._extract_basic_entities(query_text)
        
        return {
            'query_type': best_type,
            'confidence': confidence,
            'key_entities': entities,
            'reasoning': f'Basic pattern matching identified as {best_type}',
            'method': 'pattern_matching'
        }
    
    async def _analyze_with_gemini(
        self, 
        query: MedicalQuery, 
        context: Dict[str, Any], 
        basic_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced analysis using Gemini API"""
        
        analysis_prompt = f"""
        Analyze this healthcare query and provide structured analysis:
        
        Query: "{query.query_text}"
        
        Basic classification suggests: {basic_classification['query_type']} (confidence: {basic_classification['confidence']:.2f})
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Please provide analysis in JSON format:
        {{
            "query_type": "medical_term|drug_info|symptoms|document_summary|entity_extraction",
            "confidence": 0.0-1.0,
            "key_entities": ["entity1", "entity2"],
            "urgency_level": "low|medium|high|emergency",
            "reasoning": "detailed explanation of analysis",
            "suggested_tools": ["tool1", "tool2"],
            "medical_context": "relevant medical context",
            "requires_immediate_attention": boolean,
            "complexity_assessment": "simple|medium|complex"
        }}
        
        Available tools: {list(self.available_tools.keys())}
        
        Consider:
        1. Medical urgency and safety
        2. Appropriate tool selection
        3. Need for medical disclaimers
        4. Context requirements
        """
        
        try:
            response = await self.gemini_client.generate_response(
                analysis_prompt,
                context=context.get('conversation_history', []) if context else None,
                use_functions=False,
                max_tokens=800
            )
            
            # Parse JSON response
            analysis_text = response.text
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            
            if json_match:
                gemini_analysis = json.loads(json_match.group())
                
                # Validate and merge with basic classification
                return {
                    'query_type': gemini_analysis.get('query_type', basic_classification['query_type']),
                    'confidence': max(gemini_analysis.get('confidence', 0.5), basic_classification['confidence']),
                    'key_entities': gemini_analysis.get('key_entities', basic_classification['key_entities']),
                    'reasoning': gemini_analysis.get('reasoning', basic_classification['reasoning']),
                    'urgency_level': gemini_analysis.get('urgency_level', 'low'),
                    'suggested_tools': gemini_analysis.get('suggested_tools', []),
                    'medical_context': gemini_analysis.get('medical_context', ''),
                    'requires_immediate_attention': gemini_analysis.get('requires_immediate_attention', False),
                    'complexity_assessment': gemini_analysis.get('complexity_assessment', 'simple'),
                    'method': 'gemini_enhanced'
                }
            
        except Exception as e:
            logger.warning(f"Gemini analysis failed, using basic classification: {e}")
        
        # Fallback to basic classification
        return basic_classification
    
    def _extract_basic_entities(self, query_text: str) -> List[str]:
        """Extract basic entities from query text"""
        entities = []
        
        # Common medical terms
        medical_terms = [
            'diabetes', 'hypertension', 'asthma', 'cancer', 'heart disease',
            'aspirin', 'ibuprofen', 'metformin', 'insulin', 'lisinopril',
            'headache', 'fever', 'pain', 'nausea', 'fatigue'
        ]
        
        query_lower = query_text.lower()
        for term in medical_terms:
            if term in query_lower:
                entities.append(term)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', query_text)
        entities.extend(quoted_terms)
        
        # Extract capitalized terms (potential proper nouns)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', query_text)
        entities.extend(capitalized_terms[:3])  # Limit to first 3
        
        return list(set(entities))  # Remove duplicates
    
    def _assess_urgency(self, query_text: str, analysis: Dict[str, Any]) -> UrgencyLevel:
        """Assess urgency level of the query"""
        query_lower = query_text.lower()
        
        # Check for emergency keywords
        for keyword in self.emergency_keywords:
            if keyword in query_lower:
                return UrgencyLevel.EMERGENCY
        
        # Check Gemini analysis
        if analysis.get('requires_immediate_attention'):
            return UrgencyLevel.HIGH
        
        gemini_urgency = analysis.get('urgency_level', 'low')
        if gemini_urgency == 'emergency':
            return UrgencyLevel.EMERGENCY
        elif gemini_urgency == 'high':
            return UrgencyLevel.HIGH
        elif gemini_urgency == 'medium':
            return UrgencyLevel.MEDIUM
        else:
            return UrgencyLevel.LOW
    
    def _determine_tools_needed(self, query_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Determine which tools are needed for the query"""
        tools = []
        
        # Primary tool based on query type
        primary_tools = {
            'medical_term': ['search_medical_term'],
            'drug_info': ['get_drug_information'],
            'symptoms': ['analyze_symptoms'],
            'document_summary': ['summarize_medical_document'],
            'entity_extraction': ['extract_medical_entities']
        }
        
        if query_type in primary_tools:
            tools.extend(primary_tools[query_type])
        
        # Additional tools from Gemini suggestions
        suggested_tools = analysis.get('suggested_tools', [])
        for tool in suggested_tools:
            if tool in self.available_tools and tool not in tools:
                tools.append(tool)
        
        # Add complementary tools based on complexity
        if analysis.get('complexity_assessment') == 'complex':
            if 'extract_medical_entities' not in tools:
                tools.append('extract_medical_entities')
        
        return tools
    
    def _create_execution_steps(
        self, 
        tools_required: List[str], 
        analysis: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create detailed execution steps"""
        steps = []
        
        for i, tool in enumerate(tools_required):
            tool_info = self.available_tools.get(tool, {})
            
            step = {
                'step_number': i + 1,
                'tool_name': tool,
                'description': tool_info.get('description', f'Execute {tool}'),
                'estimated_time': tool_info.get('estimated_time', 2.0),
                'complexity': tool_info.get('complexity', 'simple'),
                'requires_context': tool_info.get('requires_context', False),
                'input_data': self._prepare_tool_input(tool, analysis, context),
                'expected_output': self._describe_expected_output(tool),
                'error_handling': self._define_error_handling(tool),
                'success_criteria': self._define_success_criteria(tool)
            }
            
            steps.append(step)
        
        return steps
    
    def _prepare_tool_input(self, tool: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for specific tool"""
        base_input = {
            'query_entities': analysis.get('key_entities', []),
            'medical_context': analysis.get('medical_context', ''),
            'urgency_level': analysis.get('urgency_level', 'low')
        }
        
        if tool == 'analyze_symptoms' and context:
            base_input['patient_context'] = {
                'conversation_history': context.get('conversation_history', []),
                'user_preferences': context.get('user_preferences', {})
            }
        
        return base_input
    
    def _describe_expected_output(self, tool: str) -> str:
        """Describe expected output for tool"""
        descriptions = {
            'search_medical_term': 'Medical term definition with WHO ICD information',
            'get_drug_information': 'Drug safety data including recalls and adverse events',
            'analyze_symptoms': 'General health insights with appropriate disclaimers',
            'extract_medical_entities': 'List of identified medical entities with confidence scores',
            'summarize_medical_document': 'Document summary with key findings'
        }
        
        return descriptions.get(tool, f'Output from {tool}')
    
    def _define_error_handling(self, tool: str) -> Dict[str, str]:
        """Define error handling strategy for tool"""
        return {
            'on_api_failure': 'Use fallback methods or cached data',
            'on_timeout': 'Return partial results with timeout notice',
            'on_invalid_input': 'Request input clarification from user',
            'on_no_results': 'Provide alternative suggestions'
        }
    
    def _define_success_criteria(self, tool: str) -> List[str]:
        """Define success criteria for tool execution"""
        criteria = {
            'search_medical_term': [
                'Valid medical definition retrieved',
                'Confidence score > 0.7',
                'Appropriate medical disclaimers included'
            ],
            'get_drug_information': [
                'Drug safety data retrieved',
                'Recall information processed',
                'Safety warnings included'
            ],
            'analyze_symptoms': [
                'General insights provided',
                'No diagnostic claims made',
                'Emergency recommendations if needed'
            ],
            'extract_medical_entities': [
                'Medical entities identified',
                'Confidence scores provided',
                'Entity types classified'
            ],
            'summarize_medical_document': [
                'Document summarized',
                'Key findings extracted',
                'Summary confidence > 0.6'
            ]
        }
        
        return criteria.get(tool, ['Tool executed successfully'])
    
    def _select_disclaimers(self, query_type: str, urgency_level: UrgencyLevel) -> List[str]:
        """Select appropriate medical disclaimers"""
        disclaimers = [self.disclaimer_templates['general']]
        
        if query_type == 'symptoms':
            disclaimers.append(self.disclaimer_templates['symptoms'])
        
        if query_type == 'drug_info':
            disclaimers.append(self.disclaimer_templates['drug_safety'])
        
        if urgency_level == UrgencyLevel.EMERGENCY:
            disclaimers.insert(0, self.disclaimer_templates['emergency'])
        
        disclaimers.append(self.disclaimer_templates['diagnosis'])
        
        return disclaimers
    
    def _calculate_priority(self, urgency_level: UrgencyLevel, confidence: float) -> int:
        """Calculate execution priority"""
        base_priority = {
            UrgencyLevel.EMERGENCY: 1,
            UrgencyLevel.HIGH: 2,
            UrgencyLevel.MEDIUM: 3,
            UrgencyLevel.LOW: 4
        }
        
        priority = base_priority[urgency_level]
        
        # Adjust based on confidence
        if confidence > 0.8:
            priority -= 1
        elif confidence < 0.5:
            priority += 1
        
        return max(1, min(priority, 5))  # Keep in range 1-5
    
    def _determine_context_requirements(self, tools_required: List[str]) -> Dict[str, Any]:
        """Determine what context is needed for execution"""
        requirements = {
            'conversation_history': False,
            'user_preferences': False,
            'medical_history': False,
            'current_medications': False
        }
        
        for tool in tools_required:
            tool_info = self.available_tools.get(tool, {})
            if tool_info.get('requires_context'):
                requirements['conversation_history'] = True
                
                if tool == 'analyze_symptoms':
                    requirements['medical_history'] = True
                    requirements['current_medications'] = True
        
        return requirements
    
    def _estimate_complexity(self, tools_required: List[str]) -> str:
        """Estimate overall complexity of the plan"""
        if not tools_required:
            return 'simple'
        
        complexities = []
        for tool in tools_required:
            tool_info = self.available_tools.get(tool, {})
            complexities.append(tool_info.get('complexity', 'simple'))
        
        if 'complex' in complexities:
            return 'complex'
        elif 'medium' in complexities:
            return 'medium'
        else:
            return 'simple'
    
    def _create_fallback_plan(self, query: MedicalQuery) -> QueryPlan:
        """Create fallback plan when analysis fails"""
        return QueryPlan(
            query_id=query.id,
            query_type=QueryType.MEDICAL_TERM,  # Safe default
            tools_required=['search_medical_term'],
            execution_steps=[{
                'step_number': 1,
                'tool_name': 'search_medical_term',
                'description': 'Search for medical information',
                'estimated_time': 2.0,
                'complexity': 'simple'
            }],
            priority=3,
            urgency_level=UrgencyLevel.LOW,
            confidence=0.3,
            reasoning="Fallback plan due to analysis failure",
            medical_disclaimers=[self.disclaimer_templates['general']]
        )
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get statistics about planning capabilities"""
        return {
            'available_query_types': len(self.classification_patterns),
            'available_tools': len(self.available_tools),
            'emergency_keywords': len(self.emergency_keywords),
            'disclaimer_templates': len(self.disclaimer_templates),
            'gemini_enabled': self.gemini_client is not None,
            'classification_patterns': list(self.classification_patterns.keys()),
            'tool_names': list(self.available_tools.keys())
        }