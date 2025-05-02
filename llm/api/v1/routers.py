from fastapi import APIRouter

from api.v1.endpoints.constitution_recipe import router as recipe_router
from api.v1.endpoints.constitution_diagnose import router as diagnose_router

api_router = APIRouter()

# 대화형 LLM 채팅(체질 생성) 라우터
api_router.include_router(recipe_router, prefix="/constitution_recipe", tags=["chat"])

# 체질 진단(RAG 기반) 라우터
api_router.include_router(diagnose_router, prefix="/diagnose", tags=["diagnosis"]) 