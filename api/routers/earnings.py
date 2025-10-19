"""
Earnings API Router - Endpoints for earnings calls functionality
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from workers.engine.earnings_service import EarningsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/earnings", tags=["earnings"])

# Initialize earnings service
earnings_service = EarningsService()

class EarningsQueryRequest(BaseModel):
    query: str

class EarningsQARequest(BaseModel):
    doc_id: str
    question: str

class EarningsIngestRequest(BaseModel):
    ticker: str
    quarter: str
    source_url: Optional[str] = None

@router.post("/summarize")
async def summarize_earnings(request: EarningsQueryRequest) -> Dict[str, Any]:
    """
    Summarize earnings call for a given ticker and quarter
    """
    try:
        result = earnings_service.process_earnings_query(request.query)
        return result
    except Exception as e:
        logger.error(f"Error in summarize_earnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qa")
async def earnings_qa(request: EarningsQARequest) -> Dict[str, Any]:
    """
    Answer questions about a specific earnings document
    """
    try:
        result = earnings_service.answer_earnings_question(request.doc_id, request.question)
        return result
    except Exception as e:
        logger.error(f"Error in earnings_qa: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_earnings(request: EarningsIngestRequest) -> Dict[str, Any]:
    """
    Manually ingest earnings document from URL
    """
    try:
        # This would implement manual ingestion
        # For now, return a mock response
        return {
            "ticker": request.ticker,
            "quarter": request.quarter,
            "status": "ingested",
            "doc_id": "mock_doc_id",
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in ingest_earnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{ticker}/{quarter}/status")
async def get_earnings_status(ticker: str, quarter: str) -> Dict[str, Any]:
    """
    Get status of earnings document for ticker and quarter
    """
    try:
        # This would check if document exists and return status
        # For now, return a mock response
        return {
            "ticker": ticker,
            "quarter": quarter,
            "status": "not_found",
            "available": False
        }
    except Exception as e:
        logger.error(f"Error in get_earnings_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for earnings service
    """
    return {"status": "healthy", "service": "earnings"}
