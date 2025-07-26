"""
Google Gemini API client for Intelligent Healthcare Navigator
Handles authentication, text generation, function calling, and reasoning tasks
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional, Callable
import json
from dataclasses import dataclass
from src.config import Config
from src.models import GeminiResponse
from src.utils import setup_logging

logger = setup_logging()

@dataclass
class FunctionDefinition:
    """Definition of a function that Gemini can call"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

class GeminiAPIClient:
    """Client for Google Gemini API with healthcare-specific functionality"""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini API client"""
        self.api_key = api_key or Config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Function registry for function calling
        self.functions: Dict[str, FunctionDefinition] = {}
        
        # Healthcare-specific system instructions
        self.system_instructions = self._get_healthcare_system_instructions()
        
        logger.info("Gemini API client initialized successfully")
    
    def _get_healthcare_system_instructions(self) -> str:
        """Get system instructions for healthcare context"""
        return """
You are an intelligent healthcare navigator AI assistant. Your role is to help users understand medical information clearly and directly.

CRITICAL GUIDELINES:
- You do NOT provide medical diagnoses or treatment recommendations
- Always include appropriate medical disclaimers
- Recommend consulting healthcare professionals for medical concerns
- For urgent symptoms, advise seeking immediate medical attention

RESPONSE REQUIREMENTS:
- Provide ONLY the final, user-friendly response
- Do NOT show your reasoning process or internal thinking
- Do NOT show function calls or tool execution
- Be clear, concise, and easy to understand
- Use plain language, avoiding unnecessary medical jargon
- Structure information logically with clear sections
- Include sources when available
- Always end health-related responses with appropriate disclaimers

CAPABILITIES:
- Explain medical terms and diseases in plain language
- Provide information about drug recalls and adverse events
- Offer general health insights based on symptoms (no diagnosis)
- Summarize medical documents
- Extract medical entities from text

When using functions, gather the information silently and present only the final, synthesized response to the user.
"""
    
    def register_function(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any], 
        function: Callable
    ):
        """Register a function for Gemini to call"""
        self.functions[name] = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        logger.info(f"Registered function: {name}")
    
    def _create_function_declarations(self) -> List[Dict[str, Any]]:
        """Create function declarations for Gemini API"""
        declarations = []
        
        for func_def in self.functions.values():
            declaration = {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters
            }
            declarations.append(declaration)
        
        return declarations
    
    async def generate_response(
        self, 
        prompt: str, 
        context: List[Dict[str, Any]] = None,
        use_functions: bool = True,
        max_tokens: int = 1000
    ) -> GeminiResponse:
        """Generate response using Gemini with optional function calling"""
        try:
            # Prepare the full prompt with context and system instructions
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
            
            # Prepare function tools if available and requested
            tools = None
            if use_functions and self.functions:
                function_declarations = self._create_function_declarations()
                tools = [genai.types.Tool(function_declarations=function_declarations)]
            
            # Generate response
            if tools:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    tools=tools
                )
            else:
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            
            # Process response and handle function calls
            gemini_response = await self._process_response(response, full_prompt)
            
            logger.info(f"Generated response for prompt: {prompt[:50]}...")
            return gemini_response
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            return GeminiResponse(
                text=f"I apologize, but I encountered an error processing your request. Please try again.",
                raw_response={'error': str(e)}
            )
    
    def _prepare_prompt(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Prepare the full prompt with context and instructions"""
        full_prompt_parts = [self.system_instructions]
        
        # Add conversation context if provided
        if context:
            full_prompt_parts.append("\nCONVERSATION CONTEXT:")
            for interaction in context[-5:]:  # Last 5 interactions for context
                full_prompt_parts.append(f"User: {interaction.get('query_text', '')}")
                full_prompt_parts.append(f"Assistant: {interaction.get('response_text', '')[:200]}...")
        
        # Add current prompt
        full_prompt_parts.append(f"\nCURRENT USER QUERY: {prompt}")
        
        # Add response instruction
        full_prompt_parts.append("""
Please provide a clear, helpful response to the user's query:
1. Use appropriate functions if available to gather authoritative information
2. Synthesize the information into a user-friendly response
3. Include proper medical disclaimers
4. Do NOT show your reasoning process or function calls - only provide the final response
""")
        
        return "\n".join(full_prompt_parts)
    
    async def _process_response(self, response, original_prompt: str) -> GeminiResponse:
        """Process Gemini response and handle function calls"""
        try:
            # Extract basic response data
            gemini_response = GeminiResponse.from_api_response(response)
            
            # Handle function calls if present
            if gemini_response.has_function_calls():
                function_results = []
                
                for function_call in gemini_response.function_calls:
                    function_name = function_call.get('name')
                    function_args = function_call.get('args', {})
                    
                    if function_name in self.functions:
                        try:
                            # Execute the function
                            result = await self._execute_function(function_name, function_args)
                            function_results.append({
                                'function_name': function_name,
                                'args': function_args,
                                'result': result
                            })
                            
                            logger.info(f"Executed function {function_name} with args {function_args}")
                            
                        except Exception as e:
                            logger.error(f"Error executing function {function_name}: {e}")
                            function_results.append({
                                'function_name': function_name,
                                'args': function_args,
                                'error': str(e)
                            })
                
                # Generate follow-up response with function results
                if function_results:
                    follow_up_response = await self._generate_follow_up_response(
                        original_prompt, 
                        gemini_response.text, 
                        function_results
                    )
                    
                    # Combine responses
                    gemini_response.text = follow_up_response.text
                    gemini_response.reasoning = f"Initial reasoning: {gemini_response.text}\n\nFunction results processed: {follow_up_response.reasoning}"
            
            return gemini_response
            
        except Exception as e:
            logger.error(f"Error processing Gemini response: {e}")
            return GeminiResponse(
                text="I encountered an error processing the response. Please try again.",
                raw_response={'error': str(e)}
            )
    
    async def _execute_function(self, function_name: str, args: Dict[str, Any]) -> Any:
        """Execute a registered function"""
        if function_name not in self.functions:
            raise ValueError(f"Function {function_name} not found")
        
        function_def = self.functions[function_name]
        
        # Execute function (handle both sync and async functions)
        import asyncio
        if asyncio.iscoroutinefunction(function_def.function):
            return await function_def.function(**args)
        else:
            return function_def.function(**args)
    
    async def _generate_follow_up_response(
        self, 
        original_prompt: str, 
        initial_response: str, 
        function_results: List[Dict[str, Any]]
    ) -> GeminiResponse:
        """Generate follow-up response incorporating function results"""
        
        # Prepare prompt with function results
        follow_up_prompt = f"""
Original user query: {original_prompt}

Initial analysis: {initial_response}

Function execution results:
{json.dumps(function_results, indent=2)}

Based on the function results above, please provide a comprehensive, helpful response to the user's original query. 

Requirements:
- Synthesize information from all available sources
- Provide clear, plain-language explanations
- Include appropriate medical disclaimers
- Cite sources when mentioning specific information
- If any functions returned errors, handle gracefully and use available information
"""
        
        try:
            response = self.model.generate_content(
                follow_up_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1500,
                    temperature=0.6
                )
            )
            
            return GeminiResponse.from_api_response(response)
            
        except Exception as e:
            logger.error(f"Error generating follow-up response: {e}")
            return GeminiResponse(
                text="I was able to gather some information but encountered an error synthesizing the final response.",
                raw_response={'error': str(e)}
            )
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and required actions"""
        try:
            analysis_prompt = f"""
Analyze this healthcare-related query and determine:

Query: "{query}"

Please provide a JSON response with:
{{
    "query_type": "medical_term|drug_info|symptoms|document_summary|entity_extraction",
    "confidence": 0.0-1.0,
    "key_entities": ["entity1", "entity2"],
    "urgency_level": "low|medium|high|emergency",
    "suggested_functions": ["function1", "function2"],
    "reasoning": "explanation of analysis"
}}

Available functions: {list(self.functions.keys())}
"""
            
            response = self.model.generate_content(
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3
                )
            )
            
            # Parse JSON response
            try:
                analysis = json.loads(response.text)
                logger.info(f"Query analysis completed: {analysis.get('query_type', 'unknown')}")
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return {
                    "query_type": "medical_term",
                    "confidence": 0.5,
                    "key_entities": [],
                    "urgency_level": "low",
                    "suggested_functions": [],
                    "reasoning": "Failed to parse structured analysis"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                "query_type": "unknown",
                "confidence": 0.0,
                "key_entities": [],
                "urgency_level": "low",
                "suggested_functions": [],
                "reasoning": f"Analysis error: {str(e)}"
            }
    
    def get_available_functions(self) -> List[str]:
        """Get list of available function names"""
        return list(self.functions.keys())
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            test_response = self.model.generate_content(
                "Hello, please respond with 'Connection successful'",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )
            
            success = "successful" in test_response.text.lower()
            logger.info(f"Gemini API connection test: {'PASSED' if success else 'FAILED'}")
            return success
            
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": "gemini-2.0-flash",
            "api_key_configured": bool(self.api_key),
            "functions_registered": len(self.functions),
            "available_functions": list(self.functions.keys())
        }