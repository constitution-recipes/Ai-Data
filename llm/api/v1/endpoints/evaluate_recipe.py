from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional
import requests
from core.config import AI_DATA_URL
import json
from db.session import get_recipe_db, get_chat_db
from crud.recipe import create_recipe as crud_create_recipe
from crud.chat import add_chat_message as crud_add_chat_message
import httpx
from datetime import datetime
from utils.evaluator.recipe_evaluator import evaluate_qa, evaluate_recipe
from model.recipe_model import get_recipe_llm

router = APIRouter()


# 테스트용 요청 모델 정의
class ModelTestRequest(BaseModel):
    qa_history_json: str = Field(..., description="QA 히스토리 JSON 문자열")
    recipe: str = Field(..., description="레시피 JSON 문자열")
    provider: str = Field('openai', description="AI 공급자 (openai, gemini, claude)")
    model: str = Field('gpt-3.5-turbo', description="모델 이름")
    prompt_str: str = Field(None, description="프롬프트 문자열")
    prompt_name: Optional[str] = Field(None, description="프롬프트 이름")

# 테스트 엔드포인트: AI 모델/프롬프트 실험용
@router.post("/test")
async def test_model(request: ModelTestRequest):
    print(f"[{datetime.now()}] 모델 테스트 요청: provider={request.provider}, model={request.model}")
    llm = get_recipe_llm(request.provider, request.model)
    content = llm.invoke(request.qa_history_json).content
    print(content)


    # # QA 평가
    qa_result, qa_score = evaluate_qa(content)
    # # 레시피 평가
    recipe_result, recipe_score = evaluate_recipe(request.qa_history_json, request.recipe, request.provider, request.model, request.prompt_str)
    # average_score = (qa_score + recipe_score) / 2
    # return {
    #     "provider": request.provider,
    #     "model": request.model,
    #     "prompt_str": request.prompt_str,
    #     "qa_result": qa_result,
    #     "qa_score": qa_score,
    #     "recipe_result": recipe_result,
    #     "recipe_score": recipe_score,
    #     "average_score": average_score
    # }

