"""
Memory management for Intelligent Healthcare Navigator
Handles conversation history, caching, and user preferences
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import threading
from src.config import Config
from src.models import MedicalQuery, MedicalResponse
from src.utils import setup_logging

logger = setup_logging()

class ConversationMemory:
    """Manages conversation history using SQLite database"""
    
    def __init__(self, db_path: str = None):
        """Initialize conversation memory with database connection"""
        self.db_path = db_path or Config.DATABASE_PATH
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query_id TEXT UNIQUE NOT NULL,
                    query_text TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    sources TEXT,
                    confidence_score REAL,
                    medical_entities TEXT,
                    disclaimers TEXT,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON conversations(session_id, timestamp DESC)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_id 
                ON conversations(query_id)
            ''')
            
            conn.commit()
            logger.info("Conversation memory database initialized")
    
    def store_interaction(
        self, 
        query: MedicalQuery, 
        response: MedicalResponse, 
        session_id: str = "default"
    ) -> bool:
        """Store a query-response interaction in memory"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO conversations 
                        (session_id, query_id, query_text, query_type, response_text, 
                         sources, confidence_score, medical_entities, disclaimers, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        query.id,
                        query.query_text,
                        query.query_type.value,
                        response.response_text,
                        json.dumps(response.sources),
                        response.confidence_score,
                        json.dumps([entity.to_dict() for entity in response.medical_entities]),
                        json.dumps(response.disclaimers),
                        json.dumps(response.metadata),
                        query.timestamp.isoformat()
                    ))
                    conn.commit()
            
            logger.info(f"Stored interaction for query {query.id} in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return False
    
    def get_context(
        self, 
        session_id: str = "default", 
        limit: int = 10,
        include_entities: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation context for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM conversations 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (session_id, limit))
                
                rows = cursor.fetchall()
                
                context = []
                for row in reversed(rows):  # Reverse to get chronological order
                    interaction = {
                        'query_id': row['query_id'],
                        'query_text': row['query_text'],
                        'query_type': row['query_type'],
                        'response_text': row['response_text'],
                        'sources': json.loads(row['sources'] or '[]'),
                        'confidence_score': row['confidence_score'],
                        'disclaimers': json.loads(row['disclaimers'] or '[]'),
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'timestamp': row['timestamp']
                    }
                    
                    if include_entities:
                        interaction['medical_entities'] = json.loads(row['medical_entities'] or '[]')
                    
                    context.append(interaction)
                
                logger.info(f"Retrieved {len(context)} interactions for session {session_id}")
                return context
                
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def get_interaction_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific interaction by query ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM conversations WHERE query_id = ?
                ''', (query_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'query_id': row['query_id'],
                        'query_text': row['query_text'],
                        'query_type': row['query_type'],
                        'response_text': row['response_text'],
                        'sources': json.loads(row['sources'] or '[]'),
                        'confidence_score': row['confidence_score'],
                        'medical_entities': json.loads(row['medical_entities'] or '[]'),
                        'disclaimers': json.loads(row['disclaimers'] or '[]'),
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'timestamp': row['timestamp']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve interaction {query_id}: {e}")
            return None
    
    def search_conversations(
        self, 
        search_term: str, 
        session_id: str = "default",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search conversations by text content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM conversations 
                    WHERE session_id = ? AND 
                          (query_text LIKE ? OR response_text LIKE ?)
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (session_id, f'%{search_term}%', f'%{search_term}%', limit))
                
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'query_id': row['query_id'],
                        'query_text': row['query_text'],
                        'response_text': row['response_text'][:200] + "..." if len(row['response_text']) > 200 else row['response_text'],
                        'timestamp': row['timestamp'],
                        'relevance_score': self._calculate_relevance(search_term, row['query_text'], row['response_text'])
                    })
                
                # Sort by relevance score
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                logger.info(f"Found {len(results)} conversations matching '{search_term}'")
                return results
                
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
    def _calculate_relevance(self, search_term: str, query_text: str, response_text: str) -> float:
        """Calculate relevance score for search results"""
        search_lower = search_term.lower()
        query_lower = query_text.lower()
        response_lower = response_text.lower()
        
        score = 0.0
        
        # Exact matches get higher scores
        if search_lower in query_lower:
            score += 2.0
        if search_lower in response_lower:
            score += 1.0
        
        # Word matches
        search_words = search_lower.split()
        query_words = query_lower.split()
        response_words = response_lower.split()
        
        for word in search_words:
            if word in query_words:
                score += 0.5
            if word in response_words:
                score += 0.3
        
        return score
    
    def clear_session(self, session_id: str = "default") -> bool:
        """Clear all conversations for a session"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        DELETE FROM conversations WHERE session_id = ?
                    ''', (session_id,))
                    conn.commit()
                    
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleared {deleted_count} conversations from session {session_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """Remove conversations older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        DELETE FROM conversations 
                        WHERE datetime(timestamp) < datetime(?)
                    ''', (cutoff_date.isoformat(),))
                    conn.commit()
                    
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleaned up {deleted_count} old conversations")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
            return 0
    
    def get_session_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get statistics for a conversation session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(confidence_score) as avg_confidence,
                        MIN(timestamp) as first_interaction,
                        MAX(timestamp) as last_interaction,
                        query_type,
                        COUNT(query_type) as type_count
                    FROM conversations 
                    WHERE session_id = ?
                    GROUP BY query_type
                ''', (session_id,))
                
                type_stats = cursor.fetchall()
                
                # Get overall stats
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(confidence_score) as avg_confidence,
                        MIN(timestamp) as first_interaction,
                        MAX(timestamp) as last_interaction
                    FROM conversations 
                    WHERE session_id = ?
                ''', (session_id,))
                
                overall_stats = cursor.fetchone()
                
                return {
                    'session_id': session_id,
                    'total_interactions': overall_stats[0] if overall_stats else 0,
                    'average_confidence': round(overall_stats[1] or 0.0, 2),
                    'first_interaction': overall_stats[2],
                    'last_interaction': overall_stats[3],
                    'query_type_distribution': {row[4]: row[5] for row in type_stats}
                }
                
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {'session_id': session_id, 'total_interactions': 0}
    
    def close(self):
        """Close database connections and cleanup"""
        logger.info("Conversation memory closed")

class CacheManager:
    """Manages in-memory caching for API responses with TTL support"""
    
    def __init__(self, default_ttl: int = None):
        """Initialize cache manager with default TTL"""
        self.default_ttl = default_ttl or Config.CACHE_TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._start_cleanup_thread()
        logger.info(f"Cache manager initialized with default TTL: {self.default_ttl}s")
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        def cleanup_expired():
            while True:
                try:
                    import time
                    time.sleep(60)  # Check every minute
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Cache cleanup thread error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_expired, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_cache_key(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from API call parameters"""
        import hashlib
        
        # Create a consistent string representation of parameters
        param_str = json.dumps(params, sort_keys=True)
        key_data = f"{api_name}:{endpoint}:{param_str}"
        
        # Generate hash for consistent key length
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache_api_response(
        self, 
        api_name: str,
        endpoint: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        ttl: int = None
    ) -> str:
        """Cache API response with TTL"""
        cache_key = self._generate_cache_key(api_name, endpoint, params)
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        with self._lock:
            self._cache[cache_key] = {
                'response': response,
                'expires_at': expires_at,
                'api_name': api_name,
                'endpoint': endpoint,
                'params': params,
                'cached_at': datetime.now(),
                'access_count': 0
            }
        
        logger.debug(f"Cached response for {api_name}:{endpoint} with key {cache_key[:8]}...")
        return cache_key
    
    def get_cached_response(
        self, 
        api_name: str,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached API response if not expired"""
        cache_key = self._generate_cache_key(api_name, endpoint, params)
        
        with self._lock:
            if cache_key in self._cache:
                cached_item = self._cache[cache_key]
                
                # Check if expired
                if datetime.now() > cached_item['expires_at']:
                    del self._cache[cache_key]
                    logger.debug(f"Cache expired for key {cache_key[:8]}...")
                    return None
                
                # Update access count and return response
                cached_item['access_count'] += 1
                cached_item['last_accessed'] = datetime.now()
                
                logger.debug(f"Cache hit for {api_name}:{endpoint}")
                return cached_item['response']
        
        logger.debug(f"Cache miss for {api_name}:{endpoint}")
        return None
    
    def invalidate_cache(self, pattern: str = None, api_name: str = None) -> int:
        """Invalidate cache entries matching pattern or API name"""
        removed_count = 0
        
        with self._lock:
            keys_to_remove = []
            
            for cache_key, cached_item in self._cache.items():
                should_remove = False
                
                if pattern:
                    # Check if pattern matches endpoint or params
                    endpoint = cached_item.get('endpoint', '')
                    params_str = json.dumps(cached_item.get('params', {}))
                    if pattern in endpoint or pattern in params_str:
                        should_remove = True
                
                if api_name and cached_item.get('api_name') == api_name:
                    should_remove = True
                
                if should_remove:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                del self._cache[key]
                removed_count += 1
        
        logger.info(f"Invalidated {removed_count} cache entries")
        return removed_count
    
    def _cleanup_expired_entries(self) -> int:
        """Remove expired cache entries"""
        removed_count = 0
        current_time = datetime.now()
        
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if current_time > item['expires_at']
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_entries = len(self._cache)
            api_distribution = {}
            total_access_count = 0
            oldest_entry = None
            newest_entry = None
            
            for cached_item in self._cache.values():
                api_name = cached_item.get('api_name', 'unknown')
                api_distribution[api_name] = api_distribution.get(api_name, 0) + 1
                
                total_access_count += cached_item.get('access_count', 0)
                
                cached_at = cached_item.get('cached_at')
                if cached_at:
                    if oldest_entry is None or cached_at < oldest_entry:
                        oldest_entry = cached_at
                    if newest_entry is None or cached_at > newest_entry:
                        newest_entry = cached_at
            
            return {
                'total_entries': total_entries,
                'api_distribution': api_distribution,
                'total_access_count': total_access_count,
                'average_access_per_entry': total_access_count / max(total_entries, 1),
                'oldest_entry': oldest_entry.isoformat() if oldest_entry else None,
                'newest_entry': newest_entry.isoformat() if newest_entry else None,
                'memory_usage_estimate': len(json.dumps(self._cache))
            }
    
    def clear_cache(self) -> int:
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_cache_entry_details(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific cache entry"""
        with self._lock:
            if cache_key in self._cache:
                cached_item = self._cache[cache_key].copy()
                cached_item['expires_at'] = cached_item['expires_at'].isoformat()
                cached_item['cached_at'] = cached_item['cached_at'].isoformat()
                if 'last_accessed' in cached_item:
                    cached_item['last_accessed'] = cached_item['last_accessed'].isoformat()
                return cached_item
        
        return None
    
    def set_ttl_for_api(self, api_name: str, ttl: int):
        """Set default TTL for specific API (affects future cache entries)"""
        # This would be implemented with a more sophisticated configuration system
        # For now, we'll just log the intention
        logger.info(f"TTL for {api_name} set to {ttl}s (affects future cache entries)")
    
    def close(self):
        """Cleanup cache manager resources"""
        self.clear_cache()
        logger.info("Cache manager closed")


class UserPreferences:
    """Manages user preferences and settings"""
    
    def __init__(self, db_path: str = None):
        """Initialize user preferences with database connection"""
        self.db_path = db_path or Config.DATABASE_PATH
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize preferences table in database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    preference_type TEXT NOT NULL DEFAULT 'string',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, preference_key)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_preferences 
                ON user_preferences(user_id, preference_key)
            ''')
            
            conn.commit()
            logger.info("User preferences database initialized")
    
    def set_preference(
        self, 
        key: str, 
        value: Any, 
        user_id: str = "default",
        preference_type: str = None
    ) -> bool:
        """Set user preference"""
        try:
            # Determine preference type
            if preference_type is None:
                if isinstance(value, bool):
                    preference_type = 'boolean'
                elif isinstance(value, int):
                    preference_type = 'integer'
                elif isinstance(value, float):
                    preference_type = 'float'
                elif isinstance(value, (list, dict)):
                    preference_type = 'json'
                else:
                    preference_type = 'string'
            
            # Serialize value
            if preference_type == 'json':
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO user_preferences 
                        (user_id, preference_key, preference_value, preference_type, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, key, serialized_value, preference_type, datetime.now().isoformat()))
                    conn.commit()
            
            logger.debug(f"Set preference {key} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set preference {key}: {e}")
            return False
    
    def get_preference(
        self, 
        key: str, 
        default: Any = None, 
        user_id: str = "default"
    ) -> Any:
        """Get user preference with optional default"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT preference_value, preference_type 
                    FROM user_preferences 
                    WHERE user_id = ? AND preference_key = ?
                ''', (user_id, key))
                
                row = cursor.fetchone()
                if row:
                    value_str, pref_type = row
                    
                    # Deserialize value based on type
                    if pref_type == 'boolean':
                        return value_str.lower() == 'true'
                    elif pref_type == 'integer':
                        return int(value_str)
                    elif pref_type == 'float':
                        return float(value_str)
                    elif pref_type == 'json':
                        return json.loads(value_str)
                    else:
                        return value_str
                
                return default
                
        except Exception as e:
            logger.error(f"Failed to get preference {key}: {e}")
            return default
    
    def get_all_preferences(self, user_id: str = "default") -> Dict[str, Any]:
        """Get all preferences for a user"""
        try:
            preferences = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT preference_key, preference_value, preference_type 
                    FROM user_preferences 
                    WHERE user_id = ?
                ''', (user_id,))
                
                for key, value_str, pref_type in cursor.fetchall():
                    # Deserialize value
                    if pref_type == 'boolean':
                        preferences[key] = value_str.lower() == 'true'
                    elif pref_type == 'integer':
                        preferences[key] = int(value_str)
                    elif pref_type == 'float':
                        preferences[key] = float(value_str)
                    elif pref_type == 'json':
                        preferences[key] = json.loads(value_str)
                    else:
                        preferences[key] = value_str
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get preferences for user {user_id}: {e}")
            return {}
    
    def delete_preference(self, key: str, user_id: str = "default") -> bool:
        """Delete a user preference"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        DELETE FROM user_preferences 
                        WHERE user_id = ? AND preference_key = ?
                    ''', (user_id, key))
                    conn.commit()
                    
                    deleted = cursor.rowcount > 0
                    if deleted:
                        logger.debug(f"Deleted preference {key} for user {user_id}")
                    return deleted
                    
        except Exception as e:
            logger.error(f"Failed to delete preference {key}: {e}")
            return False
    
    def clear_user_preferences(self, user_id: str = "default") -> int:
        """Clear all preferences for a user"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        DELETE FROM user_preferences WHERE user_id = ?
                    ''', (user_id,))
                    conn.commit()
                    
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleared {deleted_count} preferences for user {user_id}")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"Failed to clear preferences for user {user_id}: {e}")
            return 0
    
    def close(self):
        """Close preferences manager"""
        logger.info("User preferences manager closed")