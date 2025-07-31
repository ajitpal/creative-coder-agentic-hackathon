"""
Utility functions for Intelligent Healthcare Navigator
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from src.config import Config

def setup_logging() -> logging.Logger:
    """Set up logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configure logging
    log_filename = os.path.join(Config.LOGS_DIR, f"healthcare_navigator_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create logger
    logger = logging.getLogger('healthcare_navigator')
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks and ensure safe processing"""
    if not isinstance(text, str):
        text = str(text)
    
    if not text:
        return ""
    
    import re
    import html
    
    # HTML escape first
    text = html.escape(text)
    
    # Remove HTML tags (in case they got through)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script tags and content
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove potentially dangerous JavaScript
    js_patterns = [
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'document\.',
        r'window\.',
        r'alert\s*\(',
        r'confirm\s*\(',
        r'prompt\s*\('
    ]
    
    for pattern in js_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove SQL injection patterns
    sql_patterns = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(--|#|/\*|\*/)',
        r'(\bOR\b.*\b=\b.*\bOR\b)',
        r'(\bAND\b.*\b=\b.*\bAND\b)',
        r'(\'\s*OR\s*\')',
        r'(\"\s*OR\s*\")',
        r'(;\s*(DROP|DELETE|INSERT|UPDATE))'
    ]
    
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove command injection patterns
    cmd_patterns = [
        r'(\||&|;|\$\(|\`)',
        r'(rm\s+|del\s+|format\s+)',
        r'(wget\s+|curl\s+)',
        r'(nc\s+|netcat\s+)',
        r'(chmod\s+|chown\s+)'
    ]
    
    for pattern in cmd_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length for safety
    if len(text) > 10000:  # 10KB limit
        text = text[:10000]
    
    return text.strip()

def format_medical_disclaimer() -> str:
    """Return standard medical disclaimer text"""
    return """
⚠️  MEDICAL DISCLAIMER: This information is for educational purposes only and should not be considered medical advice. 
Always consult with qualified healthcare professionals for medical concerns. In case of emergency, contact emergency services immediately.
"""

def validate_file_type(filename: str, allowed_extensions: list = None) -> bool:
    """Validate if file type is supported"""
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.txt', '.doc', '.docx']
    
    file_extension = os.path.splitext(filename.lower())[1]
    return file_extension in allowed_extensions

def create_response_metadata(
    query_type: str,
    sources: list,
    processing_time: float,
    confidence_score: Optional[float] = None
) -> Dict[str, Any]:
    """Create standardized response metadata"""
    return {
        'query_type': query_type,
        'sources': sources,
        'processing_time': processing_time,
        'confidence_score': confidence_score,
        'timestamp': datetime.now().isoformat(),
        'disclaimer_included': True
    }

def validate_medical_query(query: str) -> tuple[bool, str]:
    """Validate medical query input"""
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string"
    
    query = query.strip()
    
    if len(query) < 3:
        return False, "Query must be at least 3 characters long"
    
    if len(query) > 5000:
        return False, "Query must be less than 5000 characters"
    
    # Check for suspicious patterns
    import re
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
        r'onload=',
        r'onerror='
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query contains potentially unsafe content"
    
    return True, "Valid query"

def validate_file_upload(filename: str, file_data: bytes, max_size: int = 10 * 1024 * 1024) -> tuple[bool, str]:
    """Validate file upload"""
    if not filename:
        return False, "Filename is required"
    
    if not file_data:
        return False, "File data is required"
    
    # Check file size
    if len(file_data) > max_size:
        return False, f"File size exceeds maximum limit of {max_size // (1024*1024)}MB"
    
    # Check file extension
    allowed_extensions = {'.pdf', '.txt', '.doc', '.docx', '.rtf'}
    import os
    file_ext = os.path.splitext(filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        return False, f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
    
    # Check for suspicious filenames
    suspicious_names = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
    if any(char in filename for char in suspicious_names):
        return False, "Filename contains invalid characters"
    
    return True, "Valid file"

def validate_user_preferences(preferences: dict) -> tuple[bool, str]:
    """Validate user preferences"""
    if not isinstance(preferences, dict):
        return False, "Preferences must be a dictionary"
    
    # Validate age
    if 'age' in preferences:
        age = preferences['age']
        if not isinstance(age, (int, float)) or age < 0 or age > 150:
            return False, "Age must be a number between 0 and 150"
    
    # Validate allergies
    if 'allergies' in preferences:
        allergies = preferences['allergies']
        if not isinstance(allergies, list):
            return False, "Allergies must be a list"
        
        for allergy in allergies:
            if not isinstance(allergy, str) or len(allergy.strip()) == 0:
                return False, "Each allergy must be a non-empty string"
            
            if len(allergy) > 100:
                return False, "Allergy names must be less than 100 characters"
    
    # Validate medical history
    if 'medical_history' in preferences:
        history = preferences['medical_history']
        if not isinstance(history, list):
            return False, "Medical history must be a list"
        
        for condition in history:
            if not isinstance(condition, str) or len(condition.strip()) == 0:
                return False, "Each medical condition must be a non-empty string"
            
            if len(condition) > 200:
                return False, "Medical condition descriptions must be less than 200 characters"
    
    return True, "Valid preferences"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    if not filename:
        return "unnamed_file"
    
    import re
    import os
    
    # Get base name and extension
    name, ext = os.path.splitext(filename)
    
    # Remove or replace dangerous characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\.+', '.', name)  # Replace multiple dots
    name = name.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Limit length
    if len(name) > 100:
        name = name[:100]
    
    # Ensure we have a name
    if not name:
        name = "file"
    
    return name + ext.lower()

def validate_api_key(api_key: str, service_name: str) -> tuple[bool, str]:
    """Validate API key format"""
    if not api_key or not isinstance(api_key, str):
        return False, f"{service_name} API key is required"
    
    api_key = api_key.strip()
    
    if len(api_key) < 10:
        return False, f"{service_name} API key appears to be too short"
    
    if len(api_key) > 200:
        return False, f"{service_name} API key appears to be too long"
    
    # Check for suspicious patterns
    if any(char in api_key for char in ['<', '>', '"', "'"]):
        return False, f"{service_name} API key contains invalid characters"
    
    return True, "Valid API key format"