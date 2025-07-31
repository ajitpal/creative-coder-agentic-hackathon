"""
Executor module for Intelligent Healthcare Navigator
Implements the Acting phase of the ReAct pattern
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import traceback # Add this line

from src.planner import QueryPlan
from src.api_manager import APIManager
from src.medical_nlp import MedicalEntityExtractor, DocumentSummarizer
from src.document_processor import DocumentProcessor
from src.models import MedicalResponse, MedicalEntity, EntityType
from src.utils import setup_logging, format_medical_disclaimer

logger = setup_logging()

@dataclass
class ExecutionResult:
    """Result of tool execution"""
    success: bool
    data: Dict[str, Any]
    tool_name: str
    execution_time: float
    confidence_score: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ToolExecutor:
    """Coordinates execution of planned actions using available tools"""
    
    def __init__(self, api_manager: APIManager = None):
        """Initialize tool executor"""
        self.api_manager = api_manager or APIManager()
        self.entity_extractor = MedicalEntityExtractor()
        self.document_summarizer = DocumentSummarizer()
        self.document_processor = DocumentProcessor()
        
        # Tool registry with metadata
        self.tools = {
            'search_medical_term': {
                'function': self.execute_medical_term_lookup,
                'type': 'api_call',
                'description': 'Search medical terms using WHO ICD API',
                'timeout': 10.0,
                'retry_count': 3
            },
            'get_drug_information': {
                'function': self.execute_drug_info_lookup,
                'type': 'api_call',
                'description': 'Get drug information from OpenFDA API',
                'timeout': 15.0,
                'retry_count': 3
            },
            'analyze_symptoms': {
                'function': self.execute_symptom_analysis,
                'type': 'nlp_processing',
                'description': 'Analyze symptoms using Gemini AI',
                'timeout': 20.0,
                'retry_count': 2
            },
            'extract_medical_entities': {
                'function': self.execute_entity_extraction,
                'type': 'nlp_processing',
                'description': 'Extract medical entities from text',
                'timeout': 10.0,
                'retry_count': 2
            },
            'summarize_medical_document': {
                'function': self.execute_document_summary,
                'type': 'document_handling',
                'description': 'Summarize medical documents',
                'timeout': 30.0,
                'retry_count': 2
            }
        }
        
        # Execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'tool_usage_count': {tool: 0 for tool in self.tools.keys()},
            'average_execution_time': 0.0
        }
        
        logger.info("Tool executor initialized with enhanced coordination")
    
    async def execute_plan(self, plan: QueryPlan, context: Dict[str, Any] = None) -> MedicalResponse:
        """Execute complete query plan"""
        logger.info(f"Executing plan for query {plan.query_id}")
        start_time = time.time()
        
        try:
            # Execute all steps
            step_results = []
            for step in plan.execution_steps:
                result = await self._execute_step(step, context, plan)
                step_results.append(result)
                
                # Early termination for emergencies
                if (plan.urgency_level.value == 'emergency' and 
                    step.get('early_termination_enabled') and 
                    result.confidence_score >= step.get('confidence_threshold', 0.7)):
                    logger.info("Early termination triggered for emergency")
                    break
            
            # Combine results into final response
            response = await self._combine_results(step_results, plan, context)
            
            execution_time = time.time() - start_time
            logger.info(f"Plan execution completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return self._create_error_response(plan, str(e))
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any], plan: QueryPlan) -> ExecutionResult:
        """Execute individual step with enhanced coordination"""
        tool_name = step.get('tool_name')
        logger.info(f"Executing step {step.get('step_number')}: {tool_name}")
        
        start_time = time.time()
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            tool_config = self.tools[tool_name]
            tool_func = tool_config['function']
            
            # Update execution statistics
            self.execution_stats['total_executions'] += 1
            self.execution_stats['tool_usage_count'][tool_name] += 1
            
            # Prepare input data
            input_data = step.get('input_data', {})
            input_data.update({
                'query_entities': plan.key_entities,
                'urgency_level': plan.urgency_level.value,
                'context': context,
                'plan_context': getattr(plan, 'context', []), # Pass context from plan
                'query_text': getattr(plan, 'query_text', '') # Pass original query text
            })
            
            # Execute tool with timeout and retry logic
            result_data = await self._execute_with_retry(
                tool_func, 
                input_data, 
                tool_config['retry_count'],
                tool_config['timeout']
            )
            
            execution_time = time.time() - start_time
            
            # Update success statistics
            self.execution_stats['successful_executions'] += 1
            self._update_average_execution_time(execution_time)
            
            return ExecutionResult(
                success=True,
                data=result_data,
                tool_name=tool_name,
                execution_time=execution_time,
                confidence_score=result_data.get('confidence_score', 0.7),
                metadata={
                    'step_number': step.get('step_number'),
                    'tool_type': tool_config['type'],
                    'retry_attempts': 0
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Step execution failed: {e}")
            logger.error(traceback.format_exc()) # Add this line to log the full traceback
            
            # Update failure statistics
            self.execution_stats['failed_executions'] += 1
            self._update_average_execution_time(execution_time)
            
            return ExecutionResult(
                success=False,
                data={},
                tool_name=tool_name,
                execution_time=execution_time,
                confidence_score=0.0,
                error=str(e),
                metadata={
                    'step_number': step.get('step_number'),
                    'tool_type': self.tools.get(tool_name, {}).get('type', 'unknown')
                }
            )
    
    async def _execute_with_retry(self, tool_func, input_data: Dict[str, Any], max_retries: int, timeout: float) -> Dict[str, Any]:
        """Execute tool function with retry logic and timeout"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(tool_func(input_data), timeout=timeout)
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Tool execution timeout on attempt {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Tool execution failed on attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        raise last_exception or Exception("Tool execution failed after all retries")
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time statistics"""
        total_executions = self.execution_stats['total_executions']
        current_avg = self.execution_stats['average_execution_time']
        
        # Calculate new average
        new_avg = ((current_avg * (total_executions - 1)) + execution_time) / total_executions
        self.execution_stats['average_execution_time'] = new_avg
    
    async def execute_tool_by_type(self, tool_type: str, input_data: Dict[str, Any]) -> List[ExecutionResult]:
        """Execute all tools of a specific type"""
        results = []
        
        for tool_name, tool_config in self.tools.items():
            if tool_config['type'] == tool_type:
                try:
                    result_data = await tool_config['function'](input_data)
                    results.append(ExecutionResult(
                        success=True,
                        data=result_data,
                        tool_name=tool_name,
                        execution_time=0.0,  # Would need to measure
                        confidence_score=result_data.get('confidence_score', 0.7)
                    ))
                except Exception as e:
                    results.append(ExecutionResult(
                        success=False,
                        data={},
                        tool_name=tool_name,
                        execution_time=0.0,
                        confidence_score=0.0,
                        error=str(e)
                    ))
        
        return results
    
    async def execute_parallel_tools(self, tool_names: List[str], input_data: Dict[str, Any]) -> List[ExecutionResult]:
        """Execute multiple tools in parallel"""
        tasks = []
        
        for tool_name in tool_names:
            if tool_name in self.tools:
                tool_config = self.tools[tool_name]
                task = asyncio.create_task(
                    self._execute_with_retry(
                        tool_config['function'],
                        input_data,
                        tool_config['retry_count'],
                        tool_config['timeout']
                    )
                )
                tasks.append((tool_name, task))
        
        results = []
        for tool_name, task in tasks:
            try:
                result_data = await task
                results.append(ExecutionResult(
                    success=True,
                    data=result_data,
                    tool_name=tool_name,
                    execution_time=0.0,  # Would need to measure individually
                    confidence_score=result_data.get('confidence_score', 0.7)
                ))
            except Exception as e:
                results.append(ExecutionResult(
                    success=False,
                    data={},
                    tool_name=tool_name,
                    execution_time=0.0,
                    confidence_score=0.0,
                    error=str(e)
                ))
        
        return results
    
    async def execute_medical_term_lookup(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute medical term lookup using WHO ICD API with Gemini simplification and fallback"""
        entities = input_data.get('query_entities', [])
        term = input_data.get('term', '')
        
        logger.debug(f"Medical term lookup input - entities: {entities}, term: '{term}'")
        logger.debug(f"Full input_data: {input_data}")
        
        if not entities and not term:
            # Try to extract a term from the query if available
            query = input_data.get('query_text', '')
            if query:
                # Extract potential terms from the query
                logger.warning("No medical terms provided for lookup - attempting to extract from query")
                
                # Remove common words and punctuation
                import re
                query_lower = query.lower()
                query_words = re.findall(r'\b[a-z]{4,}\b', query_lower)  # Find words with 4+ chars
                
                # Filter out common words
                common_words = {'what', 'when', 'where', 'which', 'who', 'why', 'how', 'about', 'tell', 'explain',
                               'information', 'know', 'learn', 'understand', 'help', 'need', 'want', 'looking',
                               'searching', 'find', 'please', 'thank', 'thanks', 'hello', 'safe', 'effective'}
                
                potential_terms = [word for word in query_words if word not in common_words]
                
                if potential_terms:
                    term = potential_terms[0]  # Use the first potential term
                    logger.info(f"Extracted potential term from query: '{term}'")
                else:
                    logger.error("No medical terms provided for lookup and none could be extracted from query")
                    raise ValueError("No medical terms provided for lookup")
            else:
                logger.error("No medical terms provided for lookup - empty entities and term, and no query text")
                raise ValueError("No medical terms provided for lookup")
        
        # Use the first entity or provided term
        primary_term = entities[0] if entities else term
        logger.info(f"Using primary term for lookup: '{primary_term}'")
        
        try:
            # First try WHO ICD API through the API manager
            who_result = await self.api_manager.search_medical_term(primary_term)
            
            if who_result.success and who_result.data:
                who_data = who_result.data
                definition = who_data.get('definition', '')
                
                # Always use Gemini to provide comprehensive medical explanation
                if self.api_manager.gemini_client:
                    # Create comprehensive prompt whether we have WHO definition or not
                    comprehensive_prompt = f"""
                    Provide a comprehensive medical explanation for: {primary_term}
                    
                    {f"WHO ICD Definition: {definition}" if definition else ""}
                    
                    Please include:
                    1. **What it is**: Clear, simple definition of the condition
                    2. **Common symptoms**: Key signs and symptoms to watch for
                    3. **Causes**: Main factors that contribute to this condition
                    4. **Risk factors**: Who is more likely to develop this condition
                    5. **When to seek care**: Clear guidance on when to see a healthcare provider
                    6. **Management**: General approaches to managing this condition
                    
                    Format the response with clear sections and bullet points.
                    Keep it factual, helpful, and under 400 words.
                    Always emphasize consulting healthcare professionals for proper diagnosis and treatment.
                    """
                    
                    try:
                        comprehensive_result = await self.api_manager.gemini_client.generate_response(comprehensive_prompt)
                        if comprehensive_result.text:
                            simplified_explanation = comprehensive_result.text
                        else:
                            simplified_explanation = definition or f"Comprehensive information about {primary_term} is being processed."
                    except Exception as e:
                        logger.warning(f"Gemini comprehensive explanation failed: {e}")
                        simplified_explanation = definition or f"Medical information about {primary_term}"
                else:
                    simplified_explanation = definition or f"Medical information about {primary_term}"
                
                return {
                    'term': primary_term,
                    'definition': simplified_explanation,
                    'technical_definition': definition,
                    'icd_code': who_data.get('code', ''),
                    'category': who_data.get('category', ''),
                    'source': 'WHO_ICD_with_AI_simplification',
                    'confidence_score': 0.9,
                    'additional_entities': entities[1:] if len(entities) > 1 else []
                }
            
            # Fallback to Gemini-only explanation if WHO API fails
            elif self.api_manager.gemini_client:
                logger.info(f"WHO API failed, using Gemini fallback for: {primary_term}")
                
                fallback_prompt = f"""
                Provide a comprehensive medical explanation for: {primary_term}
                
                Please include:
                1. **What it is**: Clear, simple definition of the condition
                2. **Common symptoms**: Key signs and symptoms to watch for
                3. **Causes**: Main factors that contribute to this condition
                4. **Risk factors**: Who is more likely to develop this condition
                5. **When to seek care**: Clear guidance on when to see a healthcare provider
                6. **Management**: General approaches to managing this condition
                
                Format the response with clear sections and bullet points.
                Keep it factual, helpful, and under 400 words.
                Always emphasize consulting healthcare professionals for proper diagnosis and treatment.
                """
                
                gemini_result = await self.api_manager.gemini_client.generate_response(fallback_prompt)
                
                if gemini_result.text:
                    return {
                        'term': primary_term,
                        'definition': gemini_result.text,
                        'source': 'AI_generated_explanation',
                        'confidence_score': 0.7,
                        'fallback_used': True,
                        'additional_entities': entities[1:] if len(entities) > 1 else []
                    }
            
            # Final fallback - return error with helpful message
            raise Exception(f'Unable to find reliable medical information for "{primary_term}". Please consult a healthcare professional for accurate information.')
            
        except Exception as e:
            logger.error(f"Medical term lookup failed for {primary_term}: {e}")
            raise Exception(f"Medical term lookup failed: {str(e)}")
    
    async def execute_drug_info_lookup(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute drug information retrieval with FDA data, recalls, and safety warnings"""
        entities = input_data.get('query_entities', [])
        drug_name = input_data.get('drug_name', '')
        context = input_data.get('context', {})
        user_allergies = context.get('user_allergies', [])
        plan_contexts = input_data.get('plan_context', [])
        
        if not entities and not drug_name:
            raise ValueError("No drug names provided for lookup")
        
        primary_drug = entities[0] if entities else drug_name
        
        try:
            # Check for allergy warnings first
            allergy_warning = None
            if user_allergies and any(allergy.lower() in primary_drug.lower() for allergy in user_allergies):
                allergy_warning = f"⚠️ ALLERGY ALERT: You indicated an allergy to {', '.join(user_allergies)}. Do not take {primary_drug} without consulting your healthcare provider."
            
            # Try to get drug information from OpenFDA API
            try:
                drug_result = await self.api_manager.get_drug_information(primary_drug)
                
                if drug_result.success and drug_result.data:
                    fda_data = drug_result.data
                    logger.debug(f"Raw FDA data for {primary_drug}: {fda_data}")
                    
                    # Log the structure of FDA data to understand available fields
                    logger.debug(f"FDA data keys: {list(fda_data.keys())}")
                    if 'adverse_events' in fda_data:
                        logger.debug(f"Adverse events data: {fda_data['adverse_events']}")
                    if 'label_info' in fda_data:
                        logger.debug(f"Label info data: {fda_data['label_info']}")
                    if 'recalls' in fda_data:
                        logger.debug(f"Recalls data: {fda_data['recalls']}")
                    
                    # Format comprehensive drug information
                    response_data = {
                        'drug_name': primary_drug,
                        'allergy_warning': allergy_warning,
                        'basic_info': {
                            'brand_names': fda_data.get('brand_names', []),
                            'generic_names': fda_data.get('generic_names', []),
                            'manufacturer': fda_data.get('manufacturer', []),
                            'indications': fda_data.get('indications_and_usage', [])
                        },
                        'safety_information': {
                            'warnings': fda_data.get('warnings', []),
                            'contraindications': fda_data.get('contraindications', []),
                            'adverse_reactions': fda_data.get('adverse_reactions', []),
                            'precautions': fda_data.get('precautions', [])
                        },
                        'recalls': {
                            'total_recalls': fda_data.get('recall_count', 0),
                            'recent_recalls': fda_data.get('recent_recalls', [])[:3],
                            'recall_summary': fda_data.get('recall_summary', '')
                        },
                        'dosage_info': {
                            'dosage_and_administration': fda_data.get('dosage_and_administration', []),
                            'how_supplied': fda_data.get('how_supplied', [])
                        },
                        'confidence_score': 0.85,
                        'source': 'OpenFDA_API',
                        'data_timestamp': fda_data.get('search_timestamp', ''),
                        'additional_entities': entities[1:] if len(entities) > 1 else []
                    }
                    
                    # Extract side effects from actual FDA data structure
                    side_effects = []
                    
                    # Try to get side effects from adverse events
                    if 'adverse_events' in fda_data and fda_data['adverse_events'].get('success'):
                        adverse_data = fda_data['adverse_events']
                        if 'summary' in adverse_data and 'top_reactions' in adverse_data['summary']:
                            top_reactions = adverse_data['summary']['top_reactions']
                            for reaction, count in top_reactions:
                                side_effects.append(f"{reaction} ({count} reports)")
                    
                    # Try to get side effects from label info
                    if 'label_info' in fda_data and fda_data['label_info'].get('success'):
                        label_data = fda_data['label_info']
                        if 'results' in label_data and label_data['results']:
                            # Look for adverse reactions in label data
                            for result in label_data['results']:
                                if 'openfda' in result and 'adverse_reactions' in result:
                                    adverse_reactions = result['openfda']['adverse_reactions']
                                    if isinstance(adverse_reactions, list):
                                        side_effects.extend(adverse_reactions[:5])  # Limit to 5
                                    elif isinstance(adverse_reactions, str):
                                        side_effects.append(adverse_reactions)
                    
                    # Add side effects to response data
                    if side_effects:
                        response_data['side_effects'] = side_effects
                        logger.debug(f"Extracted side effects: {side_effects}")
                    
                    # --- Context-specific extraction ---
                    context_sections = {
                        'pregnancy': ['pregnancy', 'pregnancy_category', 'use_in_specific_populations'],
                        'breastfeeding': ['nursing_mothers', 'lactation'],
                        'children': ['pediatric_use'],
                        'elderly': ['geriatric_use'],
                        'kidney': ['renal_impairment'],
                        'liver': ['hepatic_impairment'],
                    }
                    context_info = {}
                    logger.debug(f"Checking for contexts: {plan_contexts}")
                    if plan_contexts and 'label_info' in fda_data and fda_data['label_info'].get('results'):
                        logger.debug(f"Label info results: {fda_data['label_info']['results']}")
                        for context in plan_contexts:
                            logger.debug(f"Looking for context: {context}")
                            for section in context_sections.get(context, []):
                                logger.debug(f"Checking section: {section}")
                                for result in fda_data['label_info']['results']:
                                    logger.debug(f"Result keys: {list(result.keys())}")
                                    if section in result:
                                        logger.debug(f"Found section {section}: {result[section]}")
                                        if context not in context_info:
                                            context_info[context] = []
                                        context_info[context].append(result[section])
                    if context_info:
                        response_data['context_info'] = context_info
                        logger.debug(f"Extracted context-specific info: {context_info}")
                    else:
                        logger.debug("No context-specific info found in FDA data")
                    
                    # If context was detected but no context-specific info found, use Gemini fallback
                    if plan_contexts and not context_info:
                        logger.info(f"No context-specific info found in FDA data for {primary_drug}, using Gemini fallback")
                        if self.api_manager.gemini_client:
                            context_str = ', '.join(plan_contexts)
                            context_prompt = f"""
                            Answer this specific question: Is {primary_drug} safe for {context_str}?
                            
                            Provide a direct, focused answer that:
                            1. Directly answers whether {primary_drug} is safe for {context_str}
                            2. Explains the specific risks or safety considerations
                            3. References FDA guidelines or medical evidence
                            4. Gives clear recommendations
                            
                            Keep the answer concise (2-3 sentences) and focused specifically on {context_str} safety.
                            Do NOT provide general drug information - only answer the {context_str} safety question.
                            """
                            
                            try:
                                context_result = await self.api_manager.gemini_client.generate_response(context_prompt)
                                if context_result.text:
                                    response_data['context_fallback_info'] = {
                                        'context': plan_contexts,
                                        'answer': context_result.text
                                    }
                                    logger.debug(f"Generated context fallback info: {context_result.text}")
                            except Exception as e:
                                logger.warning(f"Gemini context fallback failed: {e}")
                    
                    return response_data
                
            except Exception as fda_error:
                logger.warning(f"OpenFDA API failed for {primary_drug}: {fda_error}")
            
            # Fallback to Gemini AI if FDA API fails or no context info found
            if self.api_manager.gemini_client:
                logger.info(f"Using Gemini fallback for drug information: {primary_drug}")
                # If context, ask Gemini specifically
                if plan_contexts:
                    context_str = ', '.join(plan_contexts)
                    drug_prompt = f"Is {primary_drug} safe for {context_str}? Please provide a concise, evidence-based answer, referencing FDA and medical guidelines."
                else:
                    drug_prompt = f"Provide comprehensive drug information for: {primary_drug}\n\nPlease include: ..."
                gemini_result = await self.api_manager.gemini_client.generate_response(drug_prompt)
                if gemini_result.text:
                    return {
                        'drug_name': primary_drug,
                        'allergy_warning': allergy_warning,
                        'comprehensive_information': gemini_result.text,
                        'confidence_score': 0.7,
                        'source': 'AI_generated_comprehensive',
                        'fallback_used': True,
                        'disclaimer': 'This information is AI-generated. Always consult healthcare professionals for medical advice.',
                        'additional_entities': entities[1:] if len(entities) > 1 else []
                    }
            
            # Ultimate fallback - return helpful error message
            raise Exception(f'Unable to find reliable drug information for "{primary_drug}". Please consult a pharmacist or healthcare provider for accurate, up-to-date information about this medication.')
                
        except Exception as e:
            logger.error(f"Drug information lookup failed for {primary_drug}: {e}")
            raise Exception(f"Drug information lookup failed: {str(e)}")
    
    async def execute_symptom_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute symptom analysis using Gemini with medical disclaimers and urgent care recommendations"""
        entities = input_data.get('query_entities', [])
        symptoms_text = input_data.get('symptoms', '')
        context = input_data.get('context', {})
        user_age = context.get('user_age', '')
        user_medical_history = context.get('medical_history', [])
        
        # Combine entities and symptoms text
        if entities:
            symptom_description = ' '.join(entities)
        elif symptoms_text:
            symptom_description = symptoms_text
        else:
            raise ValueError("No symptoms provided for analysis")
        
        try:
            # Create comprehensive symptom analysis prompt
            analysis_prompt = f"""
            Analyze these symptoms and provide helpful health insights: {symptom_description}
            
            Context:
            - User age: {user_age if user_age else 'Not specified'}
            - Medical history: {', '.join(user_medical_history) if user_medical_history else 'Not specified'}
            
            Please provide:
            1. **Possible Causes**: Common conditions that might cause these symptoms
            2. **Severity Assessment**: Is this likely urgent, moderate, or routine?
            3. **Self-Care Recommendations**: Safe things the person can do at home
            4. **When to Seek Care**: Clear guidance on when to see a doctor
            5. **Red Flags**: Symptoms that would require immediate medical attention
            
            Important guidelines:
            - Do NOT provide specific diagnoses
            - Always recommend consulting healthcare professionals
            - Include appropriate medical disclaimers
            - If symptoms suggest emergency, clearly state this
            - Keep response under 300 words
            - Be supportive but medically responsible
            """
            
            result = await self.api_manager.gemini_client.generate_response(analysis_prompt)
            
            if result.text:
                analysis_text = result.text
                
                # Check for urgency indicators
                urgency_keywords = [
                    'emergency', 'urgent', 'immediate', 'call 911', 'er', 'hospital',
                    'severe', 'critical', 'life-threatening', 'chest pain', 'difficulty breathing',
                    'stroke', 'heart attack', 'severe bleeding', 'unconscious'
                ]
                
                urgency_detected = any(keyword in analysis_text.lower() for keyword in urgency_keywords)
                
                # Determine urgency level
                if urgency_detected:
                    urgency_level = 'high'
                    urgency_message = "🚨 These symptoms may require immediate medical attention. Consider seeking emergency care."
                elif any(word in symptom_description.lower() for word in ['pain', 'fever', 'bleeding', 'nausea']):
                    urgency_level = 'moderate'
                    urgency_message = "⚠️ These symptoms should be evaluated by a healthcare provider soon."
                else:
                    urgency_level = 'low'
                    urgency_message = "ℹ️ These symptoms can typically be monitored and discussed with your healthcare provider."
                
                # Add medical disclaimers
                medical_disclaimers = [
                    "This analysis is for informational purposes only and does not constitute medical advice.",
                    "Always consult with qualified healthcare professionals for proper diagnosis and treatment.",
                    "If you're experiencing a medical emergency, call 911 or go to the nearest emergency room immediately."
                ]
                
                return {
                    'symptoms_analyzed': symptom_description,
                    'analysis': analysis_text,
                    'urgency_level': urgency_level,
                    'urgency_message': urgency_message,
                    'urgency_detected': urgency_detected,
                    'medical_disclaimers': medical_disclaimers,
                    'confidence_score': 0.75,
                    'source': 'Gemini_AI_symptom_analysis',
                    'recommendations': {
                        'seek_immediate_care': urgency_detected,
                        'monitor_symptoms': not urgency_detected,
                        'follow_up_needed': urgency_level in ['moderate', 'high']
                    },
                    'context_used': {
                        'age_considered': bool(user_age),
                        'history_considered': bool(user_medical_history)
                    }
                }
            
            # Fallback if Gemini fails
            else:
                logger.warning(f"Gemini symptom analysis failed, providing basic response")
                
                return {
                    'symptoms_analyzed': symptom_description,
                    'analysis': f"I'm unable to analyze your symptoms at this time. For symptoms like '{symptom_description}', it's best to consult with a healthcare professional who can properly evaluate your condition.",
                    'urgency_level': 'moderate',
                    'urgency_message': "⚠️ Since I cannot analyze your symptoms, please consult a healthcare provider for proper evaluation.",
                    'urgency_detected': False,
                    'medical_disclaimers': [
                        "This system is currently unable to analyze symptoms.",
                        "Please consult with qualified healthcare professionals for proper diagnosis and treatment.",
                        "If you're experiencing a medical emergency, call 911 immediately."
                    ],
                    'confidence_score': 0.3,
                    'source': 'fallback_response',
                    'system_limitation': True
                }
                
        except Exception as e:
            logger.error(f"Symptom analysis failed for '{symptom_description}': {e}")
            raise Exception(f"Symptom analysis failed: {str(e)}")
    
    async def execute_entity_extraction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute medical entity extraction with categorization and confidence scoring"""
        text = input_data.get('text', '')
        entities = input_data.get('query_entities', [])
        
        if not text and not entities:
            raise ValueError("No text or entities provided for extraction")
        
        try:
            # If we have text, use the medical NLP extractor
            if text and hasattr(self.api_manager, 'medical_nlp'):
                extraction_result = await self.api_manager.medical_nlp.extract_entities(text)
                
                if extraction_result and extraction_result.get('entities'):
                    formatted_entities = []
                    for entity in extraction_result['entities']:
                        formatted_entities.append({
                            'text': entity.get('text', ''),
                            'entity_type': entity.get('entity_type', 'unknown'),
                            'confidence': entity.get('confidence', 0.5),
                            'start_pos': entity.get('start', 0),
                            'end_pos': entity.get('end', 0),
                            'source': 'medical_nlp'
                        })
                    
                    return {
                        'text_analyzed': text,
                        'entities_found': formatted_entities,
                        'total_entities': len(formatted_entities),
                        'confidence_score': extraction_result.get('confidence_score', 0.8),
                        'source': 'medical_nlp_extraction',
                        'processing_time': extraction_result.get('processing_time', 0)
                    }
            
            # Fallback: use provided entities with enhanced categorization
            elif entities:
                medical_entities = []
                for entity_text in entities:
                    # Enhanced classification based on patterns
                    if any(term in entity_text.lower() for term in ['diabetes', 'cancer', 'asthma', 'hypertension', 'arthritis']):
                        entity_type = 'disease'
                    elif any(term in entity_text.lower() for term in ['aspirin', 'insulin', 'medication', 'drug', 'pill', 'tablet']):
                        entity_type = 'drug'
                    elif any(term in entity_text.lower() for term in ['pain', 'fever', 'headache', 'nausea', 'fatigue', 'cough']):
                        entity_type = 'symptom'
                    elif any(term in entity_text.lower() for term in ['heart', 'lung', 'kidney', 'liver', 'brain', 'stomach']):
                        entity_type = 'body_part'
                    elif any(term in entity_text.lower() for term in ['test', 'scan', 'x-ray', 'mri', 'blood work']):
                        entity_type = 'procedure'
                    else:
                        entity_type = 'medical_term'
                    
                    medical_entities.append({
                        'text': entity_text,
                        'entity_type': entity_type,
                        'confidence': 0.75,
                        'source': 'pattern_matching'
                    })
                
                return {
                    'entities_found': medical_entities,
                    'total_entities': len(medical_entities),
                    'confidence_score': 0.75,
                    'source': 'enhanced_pattern_extraction'
                }
            
            # No entities found
            else:
                return {
                    'text_analyzed': text,
                    'entities_found': [],
                    'total_entities': 0,
                    'confidence_score': 0.0,
                    'source': 'no_entities_found',
                    'message': 'No medical entities detected in the provided text'
                }
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise Exception(f"Entity extraction failed: {str(e)}")
    
    async def execute_document_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document summarization with prescription parsing and medical report analysis"""
        document_text = input_data.get('document_text', '')
        document_type = input_data.get('document_type', 'unknown')
        file_name = input_data.get('file_name', 'document')
        
        if not document_text:
            raise ValueError("No document text provided for summarization")
        
        try:
            # Use document processor for summarization
            summary_result = await self.document_processor.process_and_summarize_document(
                document_text.encode(), file_name
            )
            
            if summary_result.get('success'):
                summary_info = summary_result['summary_info']
                
                return {
                    'document_name': file_name,
                    'document_type': summary_info.get('document_type', document_type),
                    'summary': summary_info.get('summary', 'No summary available'),
                    'key_findings': summary_info.get('key_findings', []),
                    'extracted_entities': summary_info.get('extracted_entities', []),
                    'confidence_score': summary_info.get('confidence_score', 0.7),
                    'processing_time': summary_result.get('document_info', {}).get('processing_time', 0),
                    'source': 'document_processor_with_nlp',
                    'metadata': {
                        'text_length': len(document_text),
                        'extraction_method': summary_result.get('metadata', {}).get('extraction_method', 'text'),
                        'entities_count': len(summary_info.get('extracted_entities', []))
                    }
                }
            
            # Fallback to basic text analysis if document processor fails
            else:
                logger.warning("Document processor failed, using basic text analysis")
                
                # Basic text analysis
                word_count = len(document_text.split())
                sentences = document_text.split('.')
                
                # Simple key finding extraction (look for medical terms)
                medical_keywords = ['diagnosis', 'treatment', 'medication', 'prescription', 'symptoms', 'condition']
                key_findings = []
                
                for sentence in sentences[:10]:  # Check first 10 sentences
                    if any(keyword in sentence.lower() for keyword in medical_keywords):
                        key_findings.append(sentence.strip())
                
                # Basic summary (first few sentences)
                basic_summary = '. '.join(sentences[:3]).strip()
                
                return {
                    'document_name': file_name,
                    'document_type': document_type,
                    'summary': basic_summary,
                    'key_findings': key_findings[:5],  # Limit to 5 findings
                    'confidence_score': 0.5,
                    'source': 'basic_text_analysis',
                    'metadata': {
                        'word_count': word_count,
                        'sentence_count': len(sentences),
                        'processing_method': 'fallback'
                    },
                    'limitation': 'Advanced document processing unavailable, using basic text analysis'
                }
                
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            raise Exception(f"Document summarization failed: {str(e)}")
    
    async def _combine_results(
        self, 
        step_results: List[ExecutionResult], 
        plan: QueryPlan, 
        context: Dict[str, Any]
    ) -> MedicalResponse:
        """Combine step results into final medical response"""
        
        # Combine all successful results
        successful_results = [r for r in step_results if r.success]
        
        if not successful_results:
            return self._create_error_response(plan, "All execution steps failed")
        
        # Build response text
        response_parts = []
        sources = []
        entities = []
        confidence_scores = []
        
        for result in successful_results:
            if result.tool_name == 'search_medical_term':
                term = result.data.get('term', '')
                definition = result.data.get('definition', '')
                technical_definition = result.data.get('technical_definition', '')
                icd_code = result.data.get('icd_code', '')
                
                # Format comprehensive medical term explanation
                term_response = f"## {term.title()}\n\n"
                
                if definition and definition != f"Medical information about {term}":
                    term_response += f"{definition}\n\n"
                elif technical_definition:
                    term_response += f"{technical_definition}\n\n"
                else:
                    term_response += f"Medical information about {term} is being processed.\n\n"
                
                if icd_code:
                    term_response += f"**ICD-11 Code**: {icd_code}\n\n"
                
                response_parts.append(term_response.strip())
                sources.append(result.data.get('source', 'WHO_ICD'))
                
            elif result.tool_name == 'get_drug_information':
                drug_name = result.data.get('drug_name', '')
                safety_data = result.data.get('safety_data', {})
                
                # Add context-specific information FIRST if available
                context_info = result.data.get('context_info', {})
                context_fallback_info = result.data.get('context_fallback_info', {})
                
                if context_info:
                    response_parts.append(f"**{drug_name.title()} Safety Information:**")
                    for context_name, info_list in context_info.items():
                        context_display_name = {
                            'pregnancy': 'Pregnancy Safety',
                            'breastfeeding': 'Breastfeeding Safety', 
                            'children': 'Pediatric Safety',
                            'elderly': 'Geriatric Safety',
                            'kidney': 'Kidney Disease Safety',
                            'liver': 'Liver Disease Safety'
                        }.get(context_name, context_name.title())
                        
                        response_parts.append(f"**{context_display_name}:**")
                        for info in info_list:
                            if isinstance(info, str) and info.strip():
                                response_parts.append(f"• {info.strip()}")
                        response_parts.append("")  # Add spacing
                
                elif context_fallback_info:
                    # Display Gemini fallback context information prominently
                    context_str = ', '.join(context_fallback_info['context'])
                    context_display_name = {
                        'pregnancy': 'Pregnancy Safety',
                        'breastfeeding': 'Breastfeeding Safety', 
                        'children': 'Pediatric Safety',
                        'elderly': 'Geriatric Safety',
                        'kidney': 'Kidney Disease Safety',
                        'liver': 'Liver Disease Safety'
                    }.get(context_fallback_info['context'][0], context_fallback_info['context'][0].title())
                    
                    response_parts.append(f"**{drug_name.title()} - {context_display_name}:**")
                    response_parts.append(f"{context_fallback_info['answer']}")
                    response_parts.append("")  # Add spacing
                    response_parts.append("**Additional Drug Information:**")  # Separate general info
                
                # Then add general drug information
                response_parts.append(f"**Drug Information for {drug_name}**: Safety data retrieved from FDA")
                sources.append('OpenFDA')
                
                # Add allergy warning if present
                if result.data.get('allergy_warning'):
                    response_parts.append(f"⚠️ {result.data['allergy_warning']}")
                
                # Add common side effects/adverse reactions if available
                # Use the 'side_effects' field which should now contain the top_reactions
                side_effects = result.data.get('side_effects', [])
                
                if side_effects:
                    if isinstance(side_effects, list) and side_effects:
                        response_parts.append("**Common Side Effects (Adverse Reactions):**\n- " + "\n- ".join(str(r) for r in side_effects))
                    elif isinstance(side_effects, str):
                        response_parts.append(f"**Common Side Effects (Adverse Reactions):** {side_effects}")
                else:
                    # If no side effects found in FDA data, add a note
                    response_parts.append("**Side Effects**: No specific side effects data available from FDA. Please consult your healthcare provider or pharmacist for complete information.")
                
                # Add context-specific information if available
                context_info = result.data.get('context_info', {})
                if context_info:
                    response_parts.append("\n**Context-Specific Information:**")
                    for context_name, info_list in context_info.items():
                        response_parts.append(f"- **{context_name.title()}**: {', '.join(info_list)}")
                
                # Add context-specific information if available from fallback
                context_fallback_info = result.data.get('context_fallback_info', {})
                if context_fallback_info:
                    response_parts.append("\n**Context-Specific Information (Fallback):**")
                    response_parts.append(f"- **Context**: {', '.join(context_fallback_info['context'])}")
                    response_parts.append(f"- **Answer**: {context_fallback_info['answer']}")
                
            elif result.tool_name == 'analyze_symptoms':
                analysis = result.data.get('analysis', '')
                response_parts.append(f"**Symptom Analysis**: {analysis}")
                sources.append('Gemini_AI')
                
            elif result.tool_name == 'extract_medical_entities':
                entity_count = result.data.get('entities_found', 0)
                response_parts.append(f"**Medical Entities**: Found {entity_count} medical entities")
                sources.append('Medical_NLP')
                
                # Add entities to response
                for entity_dict in result.data.get('entities', []):
                    entity = MedicalEntity(
                        text=entity_dict['text'],
                        entity_type=EntityType(entity_dict['entity_type']),
                        confidence=entity_dict['confidence'],
                        source_api=entity_dict['source_api']
                    )
                    entities.append(entity)
            
            confidence_scores.append(result.confidence_score)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Combine response text
        response_text = '\n\n'.join(response_parts)
        
        # Add single medical disclaimer (avoid duplicates)
        if response_text and not any(disclaimer_text in response_text for disclaimer_text in ['MEDICAL DISCLAIMER', 'educational purposes only']):
            response_text += '\n\n' + format_medical_disclaimer().strip()
        
        return MedicalResponse(
            query_id=plan.query_id,
            response_text=response_text,
            sources=list(set(sources)),
            confidence_score=overall_confidence,
            medical_entities=entities,
            disclaimers=plan.medical_disclaimers,
            metadata={
                'execution_steps': len(step_results),
                'successful_steps': len(successful_results),
                'total_execution_time': sum(r.execution_time for r in step_results),
                'urgency_level': plan.urgency_level.value
            }
        )
    
    def _create_error_response(self, plan: QueryPlan, error_message: str) -> MedicalResponse:
        """Create error response"""
        return MedicalResponse(
            query_id=plan.query_id,
            response_text=f"I apologize, but I encountered an error processing your request: {error_message}. Please try again or rephrase your question.",
            sources=[],
            confidence_score=0.0,
            medical_entities=[],
            disclaimers=[format_medical_disclaimer()],
            metadata={'error': error_message}
        )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'available_tools': len(self.tools),
            'tool_names': list(self.tools.keys()),
            'api_manager_available': self.api_manager is not None,
            'entity_extractor_available': self.entity_extractor is not None,
            'document_processor_available': self.document_processor is not None,
            'execution_stats': self.execution_stats.copy()
        }
    
    def reset_statistics(self):
        """Reset execution statistics"""
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'tool_usage_count': {tool: 0 for tool in self.tools.keys()},
            'average_execution_time': 0.0
        }
        logger.info("Execution statistics reset")
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool"""
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not found'}
        
        tool_config = self.tools[tool_name]
        return {
            'name': tool_name,
            'type': tool_config['type'],
            'description': tool_config['description'],
            'timeout': tool_config['timeout'],
            'retry_count': tool_config['retry_count'],
            'usage_count': self.execution_stats['tool_usage_count'].get(tool_name, 0)
        }
    
    def get_tools_by_type(self, tool_type: str) -> List[str]:
        """Get list of tools by type"""
        return [
            tool_name for tool_name, config in self.tools.items()
            if config['type'] == tool_type
        ]
    
    async def validate_tool_inputs(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs for a specific tool"""
        if tool_name not in self.tools:
            return {'valid': False, 'error': f'Tool {tool_name} not found'}
        
        validation_result = {'valid': True, 'warnings': []}
        
        # Basic validation based on tool type
        tool_config = self.tools[tool_name]
        tool_type = tool_config['type']
        
        if tool_type == 'api_call':
            if not input_data.get('query_entities') and not input_data.get('term') and not input_data.get('drug_name'):
                validation_result['valid'] = False
                validation_result['error'] = 'API call tools require entities, term, or drug_name'
        
        elif tool_type == 'nlp_processing':
            if not input_data.get('text') and not input_data.get('query_entities') and not input_data.get('symptoms'):
                validation_result['valid'] = False
                validation_result['error'] = 'NLP processing tools require text, entities, or symptoms'
        
        elif tool_type == 'document_handling':
            if not input_data.get('document_text'):
                validation_result['valid'] = False
                validation_result['error'] = 'Document handling tools require document_text'
        
        return validation_result