"""
Unit tests for Gemini API client
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from src.gemini_client import GeminiAPIClient, FunctionDefinition
from src.models import GeminiResponse

class TestGeminiAPIClient:
    """Test cases for GeminiAPIClient class"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_12345"
    
    @pytest.fixture
    def client(self, mock_api_key):
        """Create GeminiAPIClient instance for testing"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_model.return_value = Mock()
                return GeminiAPIClient(api_key=mock_api_key)
    
    def test_client_initialization(self, mock_api_key):
        """Test client initialization with API key"""
        with patch('google.generativeai.configure') as mock_configure:
            with patch('google.generativeai.GenerativeModel') as mock_model:
                mock_model.return_value = Mock()
                
                client = GeminiAPIClient(api_key=mock_api_key)
                
                mock_configure.assert_called_once_with(api_key=mock_api_key)
                assert client.api_key == mock_api_key
                assert len(client.functions) == 0
    
    def test_client_initialization_without_api_key(self):
        """Test that client raises error without API key"""
        with patch('src.config.Config.GEMINI_API_KEY', None):
            with pytest.raises(ValueError, match="Gemini API key is required"):
                GeminiAPIClient()
    
    def test_register_function(self, client):
        """Test registering functions for function calling"""
        def test_function(param1: str, param2: int) -> str:
            return f"Result: {param1} - {param2}"
        
        function_name = "test_function"
        description = "A test function"
        parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            }
        }
        
        client.register_function(function_name, description, parameters, test_function)
        
        assert function_name in client.functions
        assert client.functions[function_name].name == function_name
        assert client.functions[function_name].description == description
        assert client.functions[function_name].parameters == parameters
        assert client.functions[function_name].function == test_function
    
    def test_create_function_declarations(self, client):
        """Test creating function declarations for Gemini API"""
        # Register test functions
        def func1():
            pass
        
        def func2():
            pass
        
        client.register_function("func1", "First function", {"type": "object"}, func1)
        client.register_function("func2", "Second function", {"type": "object"}, func2)
        
        declarations = client._create_function_declarations()
        
        assert len(declarations) == 2
        assert declarations[0]["name"] == "func1"
        assert declarations[0]["description"] == "First function"
        assert declarations[1]["name"] == "func2"
        assert declarations[1]["description"] == "Second function"
    
    def test_prepare_prompt_without_context(self, client):
        """Test prompt preparation without context"""
        user_prompt = "What is diabetes?"
        
        full_prompt = client._prepare_prompt(user_prompt)
        
        assert client.system_instructions in full_prompt
        assert user_prompt in full_prompt
        assert "CURRENT USER QUERY" in full_prompt
        assert "analyze this query" in full_prompt.lower()
    
    def test_prepare_prompt_with_context(self, client):
        """Test prompt preparation with conversation context"""
        user_prompt = "Tell me more about treatment"
        context = [
            {
                "query_text": "What is diabetes?",
                "response_text": "Diabetes is a metabolic disorder characterized by high blood sugar levels..."
            },
            {
                "query_text": "What are the symptoms?",
                "response_text": "Common symptoms include increased thirst, frequent urination..."
            }
        ]
        
        full_prompt = client._prepare_prompt(user_prompt, context)
        
        assert "CONVERSATION CONTEXT" in full_prompt
        assert "What is diabetes?" in full_prompt
        assert "What are the symptoms?" in full_prompt
        assert user_prompt in full_prompt
    
    @pytest.mark.asyncio
    async def test_execute_function_sync(self, client):
        """Test executing synchronous function"""
        def sync_function(name: str, age: int) -> str:
            return f"Hello {name}, you are {age} years old"
        
        client.register_function(
            "sync_test", 
            "Sync test function", 
            {"type": "object"}, 
            sync_function
        )
        
        result = await client._execute_function("sync_test", {"name": "Alice", "age": 30})
        
        assert result == "Hello Alice, you are 30 years old"
    
    @pytest.mark.asyncio
    async def test_execute_function_async(self, client):
        """Test executing asynchronous function"""
        async def async_function(message: str) -> str:
            return f"Async result: {message}"
        
        client.register_function(
            "async_test", 
            "Async test function", 
            {"type": "object"}, 
            async_function
        )
        
        result = await client._execute_function("async_test", {"message": "test"})
        
        assert result == "Async result: test"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_function(self, client):
        """Test executing non-existent function raises error"""
        with pytest.raises(ValueError, match="Function nonexistent not found"):
            await client._execute_function("nonexistent", {})
    
    @pytest.mark.asyncio
    async def test_generate_response_without_functions(self, client):
        """Test generating response without function calling"""
        mock_response = Mock()
        mock_response.text = "This is a test response about diabetes."
        mock_response.candidates = []
        
        client.model.generate_content = Mock(return_value=mock_response)
        
        response = await client.generate_response("What is diabetes?", use_functions=False)
        
        assert isinstance(response, GeminiResponse)
        assert "diabetes" in response.text.lower()
        client.model.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_functions(self, client):
        """Test generating response with function calling"""
        # Register a test function
        def search_medical_term(term: str) -> dict:
            return {"definition": f"Medical definition of {term}"}
        
        client.register_function(
            "search_medical_term",
            "Search for medical term definition",
            {
                "type": "object",
                "properties": {"term": {"type": "string"}}
            },
            search_medical_term
        )
        
        # Mock Gemini response with function call
        mock_response = Mock()
        mock_response.text = "I'll search for information about diabetes."
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].function_call = Mock()
        mock_response.candidates[0].content.parts[0].function_call.name = "search_medical_term"
        mock_response.candidates[0].content.parts[0].function_call.args = {"term": "diabetes"}
        
        # Mock follow-up response
        mock_follow_up = Mock()
        mock_follow_up.text = "Based on the search results, diabetes is a metabolic disorder..."
        
        client.model.generate_content = Mock(side_effect=[mock_response, mock_follow_up])
        
        response = await client.generate_response("What is diabetes?", use_functions=True)
        
        assert isinstance(response, GeminiResponse)
        # Should have called generate_content twice (initial + follow-up)
        assert client.model.generate_content.call_count == 2
    
    def test_analyze_query_intent(self, client):
        """Test query intent analysis"""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "query_type": "medical_term",
            "confidence": 0.95,
            "key_entities": ["diabetes"],
            "urgency_level": "low",
            "suggested_functions": ["search_medical_term"],
            "reasoning": "User is asking for definition of medical term"
        })
        
        client.model.generate_content = Mock(return_value=mock_response)
        
        analysis = client.analyze_query_intent("What is diabetes?")
        
        assert analysis["query_type"] == "medical_term"
        assert analysis["confidence"] == 0.95
        assert "diabetes" in analysis["key_entities"]
        assert analysis["urgency_level"] == "low"
    
    def test_analyze_query_intent_json_parse_error(self, client):
        """Test query intent analysis with JSON parse error"""
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        
        client.model.generate_content = Mock(return_value=mock_response)
        
        analysis = client.analyze_query_intent("What is diabetes?")
        
        # Should return fallback analysis
        assert analysis["query_type"] == "medical_term"
        assert analysis["confidence"] == 0.5
        assert "Failed to parse" in analysis["reasoning"]
    
    def test_analyze_query_intent_api_error(self, client):
        """Test query intent analysis with API error"""
        client.model.generate_content = Mock(side_effect=Exception("API Error"))
        
        analysis = client.analyze_query_intent("What is diabetes?")
        
        assert analysis["query_type"] == "unknown"
        assert analysis["confidence"] == 0.0
        assert "Analysis error" in analysis["reasoning"]
    
    def test_get_available_functions(self, client):
        """Test getting list of available functions"""
        def func1():
            pass
        
        def func2():
            pass
        
        client.register_function("func1", "Function 1", {}, func1)
        client.register_function("func2", "Function 2", {}, func2)
        
        functions = client.get_available_functions()
        
        assert len(functions) == 2
        assert "func1" in functions
        assert "func2" in functions
    
    def test_test_connection_success(self, client):
        """Test successful connection test"""
        mock_response = Mock()
        mock_response.text = "Connection successful"
        
        client.model.generate_content = Mock(return_value=mock_response)
        
        result = client.test_connection()
        
        assert result is True
        client.model.generate_content.assert_called_once()
    
    def test_test_connection_failure(self, client):
        """Test failed connection test"""
        client.model.generate_content = Mock(side_effect=Exception("Connection failed"))
        
        result = client.test_connection()
        
        assert result is False
    
    def test_get_model_info(self, client):
        """Test getting model information"""
        # Register a test function
        def test_func():
            pass
        
        client.register_function("test_func", "Test function", {}, test_func)
        
        info = client.get_model_info()
        
        assert info["model_name"] == "gemini-2.0-flash"
        assert info["api_key_configured"] is True
        assert info["functions_registered"] == 1
        assert "test_func" in info["available_functions"]
    
    def test_healthcare_system_instructions(self, client):
        """Test that healthcare system instructions are properly set"""
        instructions = client.system_instructions
        
        assert "healthcare navigator" in instructions.lower()
        assert "medical disclaimers" in instructions.lower()
        assert "do not provide medical diagnoses" in instructions.lower()
        assert "function calling" in instructions.lower()
        assert "plain language" in instructions.lower()


class TestFunctionDefinition:
    """Test cases for FunctionDefinition dataclass"""
    
    def test_function_definition_creation(self):
        """Test creating FunctionDefinition"""
        def test_func(param: str) -> str:
            return f"Result: {param}"
        
        func_def = FunctionDefinition(
            name="test_func",
            description="A test function",
            parameters={"type": "object"},
            function=test_func
        )
        
        assert func_def.name == "test_func"
        assert func_def.description == "A test function"
        assert func_def.parameters == {"type": "object"}
        assert func_def.function == test_func