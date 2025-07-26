"""
Custom exceptions for Intelligent Healthcare Navigator
Provides specific error handling for different system components
"""

class HealthcareNavigatorException(Exception):
    """Base exception for Healthcare Navigator system"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.details = details or {}

class APIException(HealthcareNavigatorException):
    """Exception for API-related errors"""
    
    def __init__(self, message: str, api_name: str, status_code: int = None, details: dict = None):
        super().__init__(message, f"API_ERROR_{api_name.upper()}", details)
        self.api_name = api_name
        self.status_code = status_code

class AuthenticationException(HealthcareNavigatorException):
    """Exception for authentication failures"""
    
    def __init__(self, message: str, service: str):
        super().__init__(message, "AUTH_ERROR", {"service": service})
        self.service = service

class RateLimitException(HealthcareNavigatorException):
    """Exception for rate limit exceeded"""
    
    def __init__(self, message: str, service: str, retry_after: int = None):
        super().__init__(message, "RATE_LIMIT_ERROR", {"service": service, "retry_after": retry_after})
        self.service = service
        self.retry_after = retry_after

class ValidationException(HealthcareNavigatorException):
    """Exception for input validation errors"""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})
        self.field = field
        self.value = value

class ProcessingException(HealthcareNavigatorException):
    """Exception for processing errors"""
    
    def __init__(self, message: str, component: str, operation: str = None):
        super().__init__(message, "PROCESSING_ERROR", {"component": component, "operation": operation})
        self.component = component
        self.operation = operation

class DocumentException(HealthcareNavigatorException):
    """Exception for document processing errors"""
    
    def __init__(self, message: str, filename: str = None, file_type: str = None):
        super().__init__(message, "DOCUMENT_ERROR", {"filename": filename, "file_type": file_type})
        self.filename = filename
        self.file_type = file_type

class MemoryException(HealthcareNavigatorException):
    """Exception for memory/storage errors"""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(message, "MEMORY_ERROR", {"operation": operation})
        self.operation = operation

class ConfigurationException(HealthcareNavigatorException):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})
        self.config_key = config_key

class TimeoutException(HealthcareNavigatorException):
    """Exception for timeout errors"""
    
    def __init__(self, message: str, operation: str, timeout_duration: float = None):
        super().__init__(message, "TIMEOUT_ERROR", {"operation": operation, "timeout_duration": timeout_duration})
        self.operation = operation
        self.timeout_duration = timeout_duration

class MedicalDataException(HealthcareNavigatorException):
    """Exception for medical data processing errors"""
    
    def __init__(self, message: str, data_type: str = None, source: str = None):
        super().__init__(message, "MEDICAL_DATA_ERROR", {"data_type": data_type, "source": source})
        self.data_type = data_type
        self.source = source

# Error handler decorator
def handle_exceptions(fallback_response=None, log_error=True):
    """Decorator for handling exceptions in methods"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HealthcareNavigatorException as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Healthcare Navigator error in {func.__name__}: {e.message}", 
                               extra={"error_code": e.error_code, "details": e.details})
                
                if fallback_response:
                    return fallback_response
                raise
            except Exception as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                
                # Wrap in our exception type
                raise ProcessingException(f"Unexpected error in {func.__name__}: {str(e)}", 
                                        component=func.__module__)
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HealthcareNavigatorException as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Healthcare Navigator error in {func.__name__}: {e.message}", 
                               extra={"error_code": e.error_code, "details": e.details})
                
                if fallback_response:
                    return fallback_response
                raise
            except Exception as e:
                if log_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                
                # Wrap in our exception type
                raise ProcessingException(f"Unexpected error in {func.__name__}: {str(e)}", 
                                        component=func.__module__)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and fallback handling"""
    
    @staticmethod
    def with_retry(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
        """Decorator for retrying failed operations"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except (APIException, TimeoutException, ProcessingException) as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            import asyncio
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            raise last_exception
                    except Exception as e:
                        # Don't retry for unexpected errors
                        raise e
                
                raise last_exception
            
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except (APIException, TimeoutException, ProcessingException) as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            raise last_exception
                    except Exception as e:
                        # Don't retry for unexpected errors
                        raise e
                
                raise last_exception
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def with_fallback(fallback_func):
        """Decorator for providing fallback functionality"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Primary function {func.__name__} failed, using fallback: {str(e)}")
                    
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
            
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Primary function {func.__name__} failed, using fallback: {str(e)}")
                    
                    return fallback_func(*args, **kwargs)
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks"""
    
    def __init__(self, operation_name: str, component: str = None, log_errors: bool = True):
        self.operation_name = operation_name
        self.component = component
        self.log_errors = log_errors
        self.logger = None
        
        if log_errors:
            import logging
            self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        if self.log_errors and self.logger:
            self.logger.error(f"Error in {self.operation_name}: {str(exc_val)}")
        
        # Convert to our exception types if needed
        if not isinstance(exc_val, HealthcareNavigatorException):
            if "timeout" in str(exc_val).lower():
                raise TimeoutException(f"Timeout in {self.operation_name}: {str(exc_val)}", 
                                     self.operation_name)
            elif "api" in str(exc_val).lower() or "http" in str(exc_val).lower():
                raise APIException(f"API error in {self.operation_name}: {str(exc_val)}", 
                                 self.component or "unknown")
            else:
                raise ProcessingException(f"Error in {self.operation_name}: {str(exc_val)}", 
                                        self.component or "unknown", self.operation_name)
        
        return False  # Re-raise the exception