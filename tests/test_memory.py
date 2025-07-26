"""
Unit tests for memory module
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from src.memory import ConversationMemory
from src.models import MedicalQuery, MedicalResponse, MedicalEntity, QueryType, EntityType

class TestConversationMemory:
    """Test cases for ConversationMemory class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def memory(self, temp_db):
        """Create ConversationMemory instance with temporary database"""
        return ConversationMemory(db_path=temp_db)
    
    @pytest.fixture
    def sample_query(self):
        """Create sample medical query"""
        return MedicalQuery(
            query_text="What is diabetes?",
            query_type=QueryType.MEDICAL_TERM
        )
    
    @pytest.fixture
    def sample_response(self, sample_query):
        """Create sample medical response"""
        entity = MedicalEntity(
            text="diabetes",
            entity_type=EntityType.DISEASE,
            confidence=0.95
        )
        
        response = MedicalResponse(
            query_id=sample_query.id,
            response_text="Diabetes is a metabolic disorder characterized by high blood sugar levels.",
            sources=["WHO_ICD", "Gemini"],
            confidence_score=0.92,
            disclaimers=["This is not medical advice"]
        )
        response.add_entity(entity)
        return response
    
    def test_database_initialization(self, temp_db):
        """Test that database is properly initialized"""
        memory = ConversationMemory(db_path=temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that tables are created
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
            assert cursor.fetchone() is not None
    
    def test_store_interaction(self, memory, sample_query, sample_response):
        """Test storing a query-response interaction"""
        session_id = "test_session"
        
        result = memory.store_interaction(sample_query, sample_response, session_id)
        
        assert result is True
        
        # Verify interaction was stored
        context = memory.get_context(session_id, limit=1)
        assert len(context) == 1
        assert context[0]['query_text'] == "What is diabetes?"
        assert context[0]['query_id'] == sample_query.id
    
    def test_get_context(self, memory, sample_query, sample_response):
        """Test retrieving conversation context"""
        session_id = "test_session"
        
        # Store multiple interactions
        for i in range(3):
            query = MedicalQuery(
                query_text=f"Test query {i}",
                query_type=QueryType.MEDICAL_TERM
            )
            response = MedicalResponse(
                query_id=query.id,
                response_text=f"Test response {i}",
                sources=["Test"]
            )
            memory.store_interaction(query, response, session_id)
        
        # Test getting limited context
        context = memory.get_context(session_id, limit=2)
        assert len(context) == 2
        
        # Test chronological order (most recent first in returned context)
        assert "Test query 2" in context[1]['query_text']
        assert "Test query 1" in context[0]['query_text']
        
        # Test getting context without entities
        context_no_entities = memory.get_context(session_id, limit=1, include_entities=False)
        assert 'medical_entities' not in context_no_entities[0] or context_no_entities[0]['medical_entities'] == []
    
    def test_get_interaction_by_id(self, memory, sample_query, sample_response):
        """Test retrieving specific interaction by ID"""
        session_id = "test_session"
        
        memory.store_interaction(sample_query, sample_response, session_id)
        
        interaction = memory.get_interaction_by_id(sample_query.id)
        
        assert interaction is not None
        assert interaction['query_id'] == sample_query.id
        assert interaction['query_text'] == "What is diabetes?"
        assert len(interaction['medical_entities']) == 1
        assert interaction['medical_entities'][0]['text'] == "diabetes"
    
    def test_search_conversations(self, memory):
        """Test searching conversations by text content"""
        session_id = "test_session"
        
        # Store conversations with different content
        queries_responses = [
            ("What is diabetes?", "Diabetes is a metabolic disorder"),
            ("Tell me about hypertension", "Hypertension is high blood pressure"),
            ("Diabetes symptoms", "Common diabetes symptoms include thirst")
        ]
        
        for query_text, response_text in queries_responses:
            query = MedicalQuery(query_text=query_text, query_type=QueryType.MEDICAL_TERM)
            response = MedicalResponse(query_id=query.id, response_text=response_text)
            memory.store_interaction(query, response, session_id)
        
        # Search for diabetes-related conversations
        results = memory.search_conversations("diabetes", session_id)
        
        assert len(results) == 2  # Should find 2 diabetes-related conversations
        assert all("diabetes" in result['query_text'].lower() or 
                  "diabetes" in result['response_text'].lower() for result in results)
        
        # Results should be sorted by relevance
        assert results[0]['relevance_score'] >= results[1]['relevance_score']
    
    def test_clear_session(self, memory, sample_query, sample_response):
        """Test clearing all conversations for a session"""
        session_id = "test_session"
        other_session_id = "other_session"
        
        # Store interactions in both sessions
        memory.store_interaction(sample_query, sample_response, session_id)
        memory.store_interaction(sample_query, sample_response, other_session_id)
        
        # Clear one session
        result = memory.clear_session(session_id)
        assert result is True
        
        # Verify session is cleared
        context = memory.get_context(session_id)
        assert len(context) == 0
        
        # Verify other session is unaffected
        other_context = memory.get_context(other_session_id)
        assert len(other_context) == 1
    
    def test_cleanup_old_conversations(self, memory):
        """Test cleaning up old conversations"""
        session_id = "test_session"
        
        # Create old query (simulate by modifying timestamp)
        old_query = MedicalQuery(
            query_text="Old query",
            query_type=QueryType.MEDICAL_TERM
        )
        old_query.timestamp = datetime.now() - timedelta(days=35)
        
        old_response = MedicalResponse(
            query_id=old_query.id,
            response_text="Old response"
        )
        
        # Create recent query
        recent_query = MedicalQuery(
            query_text="Recent query",
            query_type=QueryType.MEDICAL_TERM
        )
        
        recent_response = MedicalResponse(
            query_id=recent_query.id,
            response_text="Recent response"
        )
        
        # Store both interactions
        memory.store_interaction(old_query, old_response, session_id)
        memory.store_interaction(recent_query, recent_response, session_id)
        
        # Cleanup conversations older than 30 days
        deleted_count = memory.cleanup_old_conversations(days_to_keep=30)
        
        assert deleted_count == 1
        
        # Verify only recent conversation remains
        context = memory.get_context(session_id)
        assert len(context) == 1
        assert context[0]['query_text'] == "Recent query"
    
    def test_get_session_stats(self, memory):
        """Test getting session statistics"""
        session_id = "test_session"
        
        # Store interactions of different types
        query_types = [QueryType.MEDICAL_TERM, QueryType.DRUG_INFO, QueryType.MEDICAL_TERM]
        
        for i, query_type in enumerate(query_types):
            query = MedicalQuery(
                query_text=f"Test query {i}",
                query_type=query_type
            )
            response = MedicalResponse(
                query_id=query.id,
                response_text=f"Test response {i}",
                confidence_score=0.8 + (i * 0.1)
            )
            memory.store_interaction(query, response, session_id)
        
        stats = memory.get_session_stats(session_id)
        
        assert stats['session_id'] == session_id
        assert stats['total_interactions'] == 3
        assert stats['average_confidence'] > 0.8
        assert stats['query_type_distribution']['medical_term'] == 2
        assert stats['query_type_distribution']['drug_info'] == 1
        assert stats['first_interaction'] is not None
        assert stats['last_interaction'] is not None
    
    def test_empty_session_stats(self, memory):
        """Test getting stats for empty session"""
        stats = memory.get_session_stats("empty_session")
        
        assert stats['session_id'] == "empty_session"
        assert stats['total_interactions'] == 0
    
    def test_nonexistent_interaction(self, memory):
        """Test retrieving non-existent interaction"""
        interaction = memory.get_interaction_by_id("nonexistent_id")
        assert interaction is None
    
    def test_empty_search(self, memory):
        """Test searching in empty database"""
        results = memory.search_conversations("test", "empty_session")
        assert len(results) == 0
from src
.memory import CacheManager, UserPreferences
import time

class TestCacheManager:
    """Test cases for CacheManager class"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager instance for testing"""
        return CacheManager(default_ttl=60)  # 1 minute TTL for testing
    
    def test_cache_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager.default_ttl == 60
        assert len(cache_manager._cache) == 0
        
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 0
    
    def test_cache_api_response(self, cache_manager):
        """Test caching API responses"""
        api_name = "test_api"
        endpoint = "/search"
        params = {"query": "diabetes", "limit": 10}
        response = {"results": ["result1", "result2"]}
        
        cache_key = cache_manager.cache_api_response(api_name, endpoint, params, response)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Verify response is cached
        cached_response = cache_manager.get_cached_response(api_name, endpoint, params)
        assert cached_response == response
    
    def test_cache_key_generation(self, cache_manager):
        """Test that same parameters generate same cache key"""
        api_name = "test_api"
        endpoint = "/search"
        params1 = {"query": "diabetes", "limit": 10}
        params2 = {"limit": 10, "query": "diabetes"}  # Different order
        
        key1 = cache_manager._generate_cache_key(api_name, endpoint, params1)
        key2 = cache_manager._generate_cache_key(api_name, endpoint, params2)
        
        assert key1 == key2  # Should be same despite parameter order
    
    def test_cache_miss(self, cache_manager):
        """Test cache miss for non-existent entries"""
        cached_response = cache_manager.get_cached_response("api", "/endpoint", {"param": "value"})
        assert cached_response is None
    
    def test_cache_expiration(self, cache_manager):
        """Test cache entry expiration"""
        # Create cache manager with very short TTL
        short_ttl_cache = CacheManager(default_ttl=1)  # 1 second TTL
        
        api_name = "test_api"
        endpoint = "/test"
        params = {"test": "value"}
        response = {"data": "test"}
        
        # Cache response
        short_ttl_cache.cache_api_response(api_name, endpoint, params, response)
        
        # Should be available immediately
        cached = short_ttl_cache.get_cached_response(api_name, endpoint, params)
        assert cached == response
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        cached = short_ttl_cache.get_cached_response(api_name, endpoint, params)
        assert cached is None
    
    def test_cache_invalidation_by_pattern(self, cache_manager):
        """Test cache invalidation by pattern"""
        # Cache multiple responses
        responses = [
            ("api1", "/diabetes", {"q": "type1"}, {"result": "diabetes1"}),
            ("api1", "/diabetes", {"q": "type2"}, {"result": "diabetes2"}),
            ("api2", "/hypertension", {"q": "info"}, {"result": "hypertension"}),
        ]
        
        for api_name, endpoint, params, response in responses:
            cache_manager.cache_api_response(api_name, endpoint, params, response)
        
        # Invalidate diabetes-related entries
        removed_count = cache_manager.invalidate_cache(pattern="diabetes")
        assert removed_count == 2
        
        # Verify diabetes entries are gone
        assert cache_manager.get_cached_response("api1", "/diabetes", {"q": "type1"}) is None
        assert cache_manager.get_cached_response("api1", "/diabetes", {"q": "type2"}) is None
        
        # Verify hypertension entry remains
        assert cache_manager.get_cached_response("api2", "/hypertension", {"q": "info"}) is not None
    
    def test_cache_invalidation_by_api_name(self, cache_manager):
        """Test cache invalidation by API name"""
        # Cache responses from different APIs
        cache_manager.cache_api_response("api1", "/endpoint1", {}, {"data": "1"})
        cache_manager.cache_api_response("api1", "/endpoint2", {}, {"data": "2"})
        cache_manager.cache_api_response("api2", "/endpoint1", {}, {"data": "3"})
        
        # Invalidate all api1 entries
        removed_count = cache_manager.invalidate_cache(api_name="api1")
        assert removed_count == 2
        
        # Verify api1 entries are gone
        assert cache_manager.get_cached_response("api1", "/endpoint1", {}) is None
        assert cache_manager.get_cached_response("api1", "/endpoint2", {}) is None
        
        # Verify api2 entry remains
        assert cache_manager.get_cached_response("api2", "/endpoint1", {}) is not None
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        # Cache some responses
        cache_manager.cache_api_response("api1", "/endpoint1", {}, {"data": "1"})
        cache_manager.cache_api_response("api1", "/endpoint2", {}, {"data": "2"})
        cache_manager.cache_api_response("api2", "/endpoint1", {}, {"data": "3"})
        
        # Access some entries to increase access count
        cache_manager.get_cached_response("api1", "/endpoint1", {})
        cache_manager.get_cached_response("api1", "/endpoint1", {})
        
        stats = cache_manager.get_cache_stats()
        
        assert stats['total_entries'] == 3
        assert stats['api_distribution']['api1'] == 2
        assert stats['api_distribution']['api2'] == 1
        assert stats['total_access_count'] == 2
        assert stats['oldest_entry'] is not None
        assert stats['newest_entry'] is not None
    
    def test_clear_cache(self, cache_manager):
        """Test clearing all cache entries"""
        # Cache some responses
        cache_manager.cache_api_response("api1", "/endpoint1", {}, {"data": "1"})
        cache_manager.cache_api_response("api2", "/endpoint2", {}, {"data": "2"})
        
        # Clear cache
        cleared_count = cache_manager.clear_cache()
        assert cleared_count == 2
        
        # Verify cache is empty
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 0
    
    def test_cache_entry_details(self, cache_manager):
        """Test getting detailed cache entry information"""
        api_name = "test_api"
        endpoint = "/test"
        params = {"param": "value"}
        response = {"data": "test"}
        
        cache_key = cache_manager.cache_api_response(api_name, endpoint, params, response)
        
        details = cache_manager.get_cache_entry_details(cache_key)
        
        assert details is not None
        assert details['api_name'] == api_name
        assert details['endpoint'] == endpoint
        assert details['params'] == params
        assert details['response'] == response
        assert details['access_count'] == 0
        assert 'expires_at' in details
        assert 'cached_at' in details


class TestUserPreferences:
    """Test cases for UserPreferences class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def preferences(self, temp_db):
        """Create UserPreferences instance with temporary database"""
        return UserPreferences(db_path=temp_db)
    
    def test_preferences_initialization(self, temp_db):
        """Test preferences manager initialization"""
        preferences = UserPreferences(db_path=temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that preferences table is created
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
            )
            assert cursor.fetchone() is not None
    
    def test_set_and_get_string_preference(self, preferences):
        """Test setting and getting string preferences"""
        user_id = "test_user"
        key = "language"
        value = "english"
        
        result = preferences.set_preference(key, value, user_id)
        assert result is True
        
        retrieved_value = preferences.get_preference(key, user_id=user_id)
        assert retrieved_value == value
    
    def test_set_and_get_boolean_preference(self, preferences):
        """Test setting and getting boolean preferences"""
        user_id = "test_user"
        key = "show_disclaimers"
        value = True
        
        result = preferences.set_preference(key, value, user_id)
        assert result is True
        
        retrieved_value = preferences.get_preference(key, user_id=user_id)
        assert retrieved_value is True
        assert isinstance(retrieved_value, bool)
    
    def test_set_and_get_integer_preference(self, preferences):
        """Test setting and getting integer preferences"""
        user_id = "test_user"
        key = "max_results"
        value = 10
        
        result = preferences.set_preference(key, value, user_id)
        assert result is True
        
        retrieved_value = preferences.get_preference(key, user_id=user_id)
        assert retrieved_value == 10
        assert isinstance(retrieved_value, int)
    
    def test_set_and_get_json_preference(self, preferences):
        """Test setting and getting JSON preferences"""
        user_id = "test_user"
        key = "favorite_topics"
        value = ["diabetes", "hypertension", "cardiology"]
        
        result = preferences.set_preference(key, value, user_id)
        assert result is True
        
        retrieved_value = preferences.get_preference(key, user_id=user_id)
        assert retrieved_value == value
        assert isinstance(retrieved_value, list)
    
    def test_get_preference_with_default(self, preferences):
        """Test getting non-existent preference with default value"""
        default_value = "default_language"
        
        retrieved_value = preferences.get_preference(
            "nonexistent_key", 
            default=default_value, 
            user_id="test_user"
        )
        
        assert retrieved_value == default_value
    
    def test_get_all_preferences(self, preferences):
        """Test getting all preferences for a user"""
        user_id = "test_user"
        
        # Set multiple preferences
        test_preferences = {
            "language": "english",
            "show_disclaimers": True,
            "max_results": 20,
            "topics": ["diabetes", "cardiology"]
        }
        
        for key, value in test_preferences.items():
            preferences.set_preference(key, value, user_id)
        
        # Get all preferences
        all_prefs = preferences.get_all_preferences(user_id)
        
        assert len(all_prefs) == 4
        assert all_prefs["language"] == "english"
        assert all_prefs["show_disclaimers"] is True
        assert all_prefs["max_results"] == 20
        assert all_prefs["topics"] == ["diabetes", "cardiology"]
    
    def test_delete_preference(self, preferences):
        """Test deleting a preference"""
        user_id = "test_user"
        key = "temp_setting"
        value = "temporary"
        
        # Set preference
        preferences.set_preference(key, value, user_id)
        assert preferences.get_preference(key, user_id=user_id) == value
        
        # Delete preference
        result = preferences.delete_preference(key, user_id)
        assert result is True
        
        # Verify it's deleted
        retrieved_value = preferences.get_preference(key, default="not_found", user_id=user_id)
        assert retrieved_value == "not_found"
    
    def test_clear_user_preferences(self, preferences):
        """Test clearing all preferences for a user"""
        user_id = "test_user"
        other_user_id = "other_user"
        
        # Set preferences for both users
        preferences.set_preference("setting1", "value1", user_id)
        preferences.set_preference("setting2", "value2", user_id)
        preferences.set_preference("setting1", "other_value", other_user_id)
        
        # Clear preferences for test_user
        cleared_count = preferences.clear_user_preferences(user_id)
        assert cleared_count == 2
        
        # Verify test_user preferences are cleared
        all_prefs = preferences.get_all_preferences(user_id)
        assert len(all_prefs) == 0
        
        # Verify other_user preferences remain
        other_prefs = preferences.get_all_preferences(other_user_id)
        assert len(other_prefs) == 1
        assert other_prefs["setting1"] == "other_value"
    
    def test_preference_update(self, preferences):
        """Test updating existing preference"""
        user_id = "test_user"
        key = "theme"
        
        # Set initial value
        preferences.set_preference(key, "light", user_id)
        assert preferences.get_preference(key, user_id=user_id) == "light"
        
        # Update value
        preferences.set_preference(key, "dark", user_id)
        assert preferences.get_preference(key, user_id=user_id) == "dark"
        
        # Verify only one entry exists
        all_prefs = preferences.get_all_preferences(user_id)
        assert len(all_prefs) == 1
    
    def test_multiple_users(self, preferences):
        """Test preferences isolation between users"""
        user1 = "user1"
        user2 = "user2"
        key = "language"
        
        # Set different values for different users
        preferences.set_preference(key, "english", user1)
        preferences.set_preference(key, "spanish", user2)
        
        # Verify isolation
        assert preferences.get_preference(key, user_id=user1) == "english"
        assert preferences.get_preference(key, user_id=user2) == "spanish"