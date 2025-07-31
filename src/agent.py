"""
Main Agent Controller for Intelligent Healthcare Navigator
Orchestrates the complete ReAct workflow: Reasoning -> Acting -> Observation
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.models import MedicalQuery, MedicalResponse, QueryType
from src.planner import QueryPlanner
from src.context_planner import ContextAwarePlanner
from src.executor import ToolExecutor, ExecutionResult
from src.memory import ConversationMemory, CacheManager, UserPreferences
from src.api_manager import APIManager
from src.gemini_client import GeminiAPIClient
from src.document_processor import DocumentProcessor
from src.config import Config
from src.utils import setup_logging, sanitize_input

logger = setup_logging()

@dataclass
class AgentResponse:
    """Complete agent response with metadata"""
    response: MedicalResponse
    execution_result: ExecutionResult
    conversation_stored: bool
    processing_time: float

class HealthcareNavigatorAgent:
    """Main agent that orchestrates the complete healthcare navigation workflow"""
    
    def __init__(self, session_id: str = "default"):
        """Initialize the healthcare navigator agent"""
        self.session_id = session_id
        
        # Initialize core components
        self._initialize_components()
        
        logger.info(f"Healthcare Navigator Agent initialized for session {session_id}")
    
    def _initialize_components(self):
        """Initialize all agent components"""
        try:
            # Memory components
            self.conversation_memory = ConversationMemory()
            self.cache_manager = CacheManager()
            self.user_preferences = UserPreferences()
            
            # API components
            self.gemini_client = GeminiAPIClient() if Config.GEMINI_API_KEY else None
            self.api_manager = APIManager(self.cache_manager)
            
            # Planning components
            self.query_planner = QueryPlanner(self.gemini_client)
            self.context_planner = ContextAwarePlanner(self.query_planner, self.conversation_memory)
            
            # Execution components
            self.executor = ToolExecutor(self.api_manager)
            
            # Document processing
            self.document_processor = DocumentProcessor()
            
            logger.info("All agent components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent components: {e}")
            raise
    
    async def process_query(self, query_text: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process user query through complete ReAct workflow"""
        import time
        start_time = time.time()
        
        try:
            # Sanitize input
            query_text = sanitize_input(query_text)
            logger.info(f"Processing query: {query_text[:100]}...")
            
            # Create medical query
            medical_query = MedicalQuery(query_text=query_text)
            
            # Get conversation context
            full_context = await self._prepare_context(context)
            
            # REASONING PHASE: Create execution plan
            logger.info("🧠 REASONING: Analyzing query and creating plan...")
            if self.context_planner and full_context.get('conversation_history'):
                plan = await self.context_planner.create_context_aware_plan(medical_query, full_context)
            else:
                plan = await self.query_planner.analyze_query(medical_query, full_context)
            
            logger.info(f"Plan created: {plan.query_type.value} with {len(plan.tools_required)} tools")
            
            # ACTING PHASE: Execute the plan
            logger.info("⚡ ACTING: Executing plan with healthcare tools...")
            medical_response = await self.executor.execute_plan(plan, full_context)
            
            # OBSERVATION PHASE: Store results and learn
            logger.info("OBSERVATION: Storing results and updating memory...")
            conversation_stored = self.conversation_memory.store_interaction(
                medical_query, medical_response, self.session_id
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            # Create execution result for consistency
            execution_result = ExecutionResult(
                success=True,
                data={'medical_response': medical_response.to_dict()},
                tool_name='plan_executor',
                execution_time=processing_time,
                confidence_score=medical_response.confidence_score,
                metadata={'plan_id': plan.query_id}
            )
            
            return AgentResponse(
                response=medical_response,
                execution_result=execution_result,
                conversation_stored=conversation_stored,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Create fallback response
            fallback_response = MedicalResponse(
                query_id="error",
                response_text=f"I apologize, but I encountered an error processing your request. Please try again or contact support. Error: {str(e)}",
                sources=[],
                disclaimers=["This system encountered a technical error. Please consult healthcare professionals for medical advice."]
            )
            
            return AgentResponse(
                response=fallback_response,
                execution_result=ExecutionResult(
                    success=False,
                    data={'error': 'Query processing failed'},
                    tool_name='error_handler',
                    execution_time=0.0,
                    confidence_score=0.0,
                    error='Query processing failed',
                    metadata={'error': str(e)}
                ),
                conversation_stored=False,
                processing_time=time.time() - start_time
            )
    
    async def handle_document_upload(self, file_data: bytes, filename: str, file_size: int = None) -> AgentResponse:
        """Handle document upload and processing"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Processing document upload: {filename}")
            
            # Process and summarize document
            result = await self.document_processor.process_and_summarize_document(
                file_data, filename, file_size
            )
            
            if not result['success']:
                error_response = MedicalResponse(
                    query_id="doc_error",
                    response_text=f"Failed to process document: {result['error']}",
                    sources=[],
                    disclaimers=["Document processing failed. Please ensure the file is a supported format."]
                )
                
                return AgentResponse(
                    response=error_response,
                    execution_result=ExecutionResult(
                        success=False,
                        data={'error': result['error']},
                        tool_name='document_processor',
                        execution_time=time.time() - start_time,
                        confidence_score=0.0,
                        error=result['error']
                    ),
                    conversation_stored=False,
                    processing_time=time.time() - start_time
                )
            
            # Create response from document processing
            summary_info = result['summary_info']
            doc_info = result['document_info']
            
            response_text = f"""**Document Processed: {filename}**

**Document Type**: {summary_info.get('document_type', 'Unknown')}

**Summary**: {summary_info.get('summary', 'No summary available')}

**Key Findings**:
{chr(10).join(f"• {finding}" for finding in summary_info.get('key_findings', []))}

**Extracted Entities**: {len(summary_info.get('extracted_entities', []))} medical entities identified

**Processing Details**:
• File size: {doc_info.get('file_size', 0)} bytes
• Text extracted: {doc_info.get('text_length', 0)} characters
• Processing time: {doc_info.get('processing_time', 0):.2f} seconds"""
            
            medical_response = MedicalResponse(
                query_id="document_upload",
                response_text=response_text,
                sources=["document_processor", "medical_nlp"],
                confidence_score=summary_info.get('confidence_score', 0.7),
                medical_entities=[],  # Could add extracted entities here
                disclaimers=["This document analysis is for informational purposes only. Consult healthcare professionals for medical decisions."],
                metadata={
                    'document_type': summary_info.get('document_type'),
                    'processing_method': result['metadata'].get('extraction_method'),
                    'file_info': doc_info
                }
            )
            
            # Store in conversation memory
            doc_query = MedicalQuery(
                query_text=f"Document uploaded: {filename}",
                query_type=QueryType.DOCUMENT_SUMMARY
            )
            
            conversation_stored = self.conversation_memory.store_interaction(
                doc_query, medical_response, self.session_id
            )
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                response=medical_response,
                execution_result=ExecutionResult(
                    success=True,
                    data=result,
                    tool_name='document_processor',
                    execution_time=processing_time,
                    confidence_score=summary_info.get('confidence_score', 0.7),
                    metadata=result['metadata']
                ),
                conversation_stored=conversation_stored,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Document upload processing failed: {e}")
            
            error_response = MedicalResponse(
                query_id="doc_error",
                response_text=f"Failed to process document upload: {str(e)}",
                disclaimers=["Document processing encountered an error. Please try again with a different file."]
            )
            
            return AgentResponse(
                response=error_response,
                execution_result=ExecutionResult(
                    success=False,
                    data={'error': str(e)},
                    tool_name='document_processor',
                    execution_time=time.time() - start_time,
                    confidence_score=0.0,
                    error=str(e)
                ),
                conversation_stored=False,
                processing_time=time.time() - start_time
            )
    
    async def _prepare_context(self, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare complete context for query processing"""
        context = {
            'session_id': self.session_id,
            'conversation_history': self.conversation_memory.get_context(self.session_id, limit=10),
            'user_preferences': self.user_preferences.get_all_preferences(self.session_id)
        }
        
        if additional_context:
            context.update(additional_context)
        
        return context
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history for current session"""
        return self.conversation_memory.get_context(self.session_id, limit=limit)
    
    def clear_conversation_history(self) -> bool:
        """Clear conversation history for current session"""
        return self.conversation_memory.clear_session(self.session_id)
    
    def set_user_preference(self, key: str, value: Any) -> bool:
        """Set user preference"""
        return self.user_preferences.set_preference(key, value, self.session_id)
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        return self.user_preferences.get_preference(key, default, self.session_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Check API health
            api_health = await self.api_manager.health_check()
            
            # Get memory stats
            session_stats = self.conversation_memory.get_session_stats(self.session_id)
            cache_stats = self.cache_manager.get_cache_stats()
            
            # Get component status
            component_status = {
                'gemini_client': self.gemini_client is not None,
                'api_manager': True,
                'conversation_memory': True,
                'cache_manager': True,
                'query_planner': True,
                'context_planner': self.context_planner is not None,
                'executor': True,
                'document_processor': True
            }
            
            return {
                'session_id': self.session_id,
                'system_healthy': api_health.get('overall', {}).get('healthy', False),
                'api_health': api_health,
                'session_stats': session_stats,
                'cache_stats': cache_stats,
                'component_status': component_status
            }
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {
                'session_id': self.session_id,
                'system_healthy': False,
                'error': str(e),
                'component_status': {
                    'gemini_client': self.gemini_client is not None,
                    'api_manager': True,
                    'conversation_memory': True,
                    'cache_manager': True,
                    'query_planner': True,
                    'context_planner': self.context_planner is not None,
                    'executor': True,
                    'document_processor': True
                }
            }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'api_manager') and self.api_manager:
                # Close API connections if needed
                pass
            logger.info(f"Healthcare Navigator Agent cleaned up for session {self.session_id}")
        except:
            pass