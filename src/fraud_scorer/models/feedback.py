# src/fraud_scorer/models/feedback.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FeedbackItem(BaseModel):
    field: str
    originalValue: Optional[str] = None
    newValue: Optional[str] = None
    status: str  # 'confirmed' o 'corrected'

class FeedbackPayload(BaseModel):
    feedback: List[FeedbackItem]