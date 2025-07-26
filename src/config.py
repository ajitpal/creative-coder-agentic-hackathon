"""
Configuration management for Intelligent Healthcare Navigator
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
    WHO_ICD_API_KEY: Optional[str] = os.getenv('WHO_ICD_API_KEY')
    OPENFDA_API_KEY: Optional[str] = os.getenv('OPENFDA_API_KEY')
    METRIPORT_API_KEY: Optional[str] = os.getenv('METRIPORT_API_KEY')
    
    # WHO ICD OAuth2 Credentials
    WHO_ICD_CLIENT_ID: str = os.getenv('WHO_ICD_CLIENT_ID')
    WHO_ICD_CLIENT_SECRET: str = os.getenv('WHO_ICD_CLIENT_SECRET')
    WHO_ICD_TOKEN_URL: str = "https://icdaccessmanagement.who.int/connect/token"
    
    # Application Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    MAX_CONVERSATION_HISTORY: int = int(os.getenv('MAX_CONVERSATION_HISTORY', '50'))
    
    # API Endpoints
    WHO_ICD_BASE_URL: str = "https://id.who.int/icd/release/11/2019-04/mms"
    OPENFDA_BASE_URL: str = "https://api.fda.gov"
    
    # File paths
    DATABASE_PATH: str = "data/healthcare_navigator.db"
    LOGS_DIR: str = "logs"
    CACHE_DIR: str = "cache"
    
    @classmethod
    def validate_required_keys(cls) -> list:
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not cls.GEMINI_API_KEY:
            missing_keys.append('GEMINI_API_KEY')
        
        if not cls.WHO_ICD_API_KEY:
            missing_keys.append('WHO_ICD_API_KEY')
            
        return missing_keys
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if minimum required configuration is present"""
        return len(cls.validate_required_keys()) == 0