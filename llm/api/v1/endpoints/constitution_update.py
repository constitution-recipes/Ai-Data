from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import core.config as config

router = APIRouter()

# MongoDB 연결
mongo_client = MongoClient(config.settings.MONGODB_URI)
db = mongo_client[config.settings.MONGODB_DB_NAME]
users_col = db['users']

# 요청/응답 모델
class UpdateRequest(BaseModel):
    constitution: str
    reason: str
    confidence: float

class UpdateResponse(BaseModel):
    user_id: str
    updated: bool

@router.put("/{user_id}/constitution", response_model=UpdateResponse)
async def update_constitution(user_id: str, req: UpdateRequest):
    try:
        oid = ObjectId(user_id)
        update_fields = {
            "constitution": req.constitution,
            "diagnosis_reason": req.reason,
            "confidence": req.confidence,
            "diagnosis_date": datetime.utcnow(),
        }
        result = users_col.update_one({"_id": oid}, {"$set": update_fields})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return UpdateResponse(user_id=user_id, updated=result.modified_count > 0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 