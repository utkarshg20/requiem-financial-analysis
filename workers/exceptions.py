"""Standardized error handling for Requiem API"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class RequiemError(Exception):
    """Base exception for Requiem API"""
    def __init__(self, message: str, code: str, details: Dict[str, Any] = None, trace_id: str = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.trace_id = trace_id
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "trace_id": self.trace_id
        }


class ValidationError(RequiemError):
    """Input validation errors (422)"""
    def __init__(self, message: str, details: Dict[str, Any] = None, trace_id: str = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            trace_id=trace_id
        )


class DataError(RequiemError):
    """Data-related errors (422)"""
    def __init__(self, message: str, details: Dict[str, Any] = None, trace_id: str = None):
        super().__init__(
            message=message,
            code="DATA_ERROR",
            details=details,
            trace_id=trace_id
        )


class InternalError(RequiemError):
    """Internal server errors (500)"""
    def __init__(self, message: str, details: Dict[str, Any] = None, trace_id: str = None):
        super().__init__(
            message=message,
            code="INTERNAL_ERROR",
            details=details,
            trace_id=trace_id
        )


class ExternalAPIError(RequiemError):
    """External API errors (502)"""
    def __init__(self, message: str, details: Dict[str, Any] = None, trace_id: str = None):
        super().__init__(
            message=message,
            code="EXTERNAL_API_ERROR",
            details=details,
            trace_id=trace_id
        )
