from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import openai
import json
from datetime import datetime
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
import config
import requests
import httpx  # 비동기 HTTP 호출용

# 환경 변수 설정
os.environ['OPENAI_API_KEY'] = config.settings.OPENAI_API_KEY
openai.api_key = config.settings.OPENAI_API_KEY

# MongoDB 연결 설정
mongo_client = MongoClient(config.settings.MONGODB_URI)
db = mongo_client.get_database(config.settings.MONGODB_DB_NAME)
collection = db.chat_sessions

# config에서 관리하는 백엔드 URL 사용
BACKEND_URL = config.settings.BACKEND_URL

app = FastAPI(
    title="AI-Data LLM Service",
    description="Microservice for LLM chat using OpenAI or LangChain",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    feature: Optional[str] = None
    messages: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]

class ChatResponse(BaseModel):
    message: str
    is_recipe: bool = False

# ChatOpenAI에 openai_api_key 파라미터로 전달하여 OpenAI 클라이언트에 API 키 설정
llm = ChatOpenAI(
    model=config.settings.MODEL_NAME,
    openai_api_key=config.settings.OPENAI_API_KEY
)

# PydanticOutputParser 설정: 하나의 Recipe만 반환
class Recipe(BaseModel):
    title: str = Field(..., description="레시피 제목")
    description: str = Field(..., description="레시피 설명")
    difficulty: str = Field(..., description="난이도")
    cookTime: str = Field(..., description="조리 시간")
    ingredients: list[str] = Field(..., description="재료 목록")
    image: str = Field(..., description="이미지 URL")
    rating: float = Field(..., description="평점")
    suitableFor: str = Field(..., description="적합 대상 설명")
    reason: str = Field(..., description="레시피 생성 이유 설명")
    suitableBodyTypes: list[str] = Field(..., description="이 음식에 적합한 체질 리스트 (목양체질, 목음체질, 토양체질, 토음체질, 금양체질, 금음체질, 수양체질, 수음체질)")
    tags: list[str] = Field(..., description="태그 목록")
    steps: list[str] = Field(..., description="조리 단계 리스트")
    servings: str = Field(..., description="인분 정보")
    nutritionalInfo: str = Field(..., description="영양 정보")

parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = parser.get_format_instructions()

@app.post("/api/v1/users/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 대화형 챗봇 역할, 요구사항 수집 흐름 및 JSON 출력 방식 정의
        base_prompt = (
            "너는 한의학 기반 체질 의학 전문가이자 대화형 챗봇이야. "
            "체질, 알레르기, 건강 상태, 식이 제한, 원하는 음식, 조리 도구, 인원수 등을 차례대로 물어보고, "
            "사용자가 '네'라고 답하기 전까지는 레시피를 제공하지 말고 반드시 추가 질문을 이어가. "
            "사용자가 '네'라고 답하면, 지금까지 수집된 모든 요구사항을 바탕으로 하나의 레시피를 생성해. "
            "출력 형식 지침(format_instructions)에 따라 하나의 JSON 객체만 반환하고, "
            "절대로 Markdown, 코드펜스, 설명 문구 없이 순수 JSON만 출력해야 해."
        )
        system_message = {"role": "system", "content": base_prompt}
        # output parser 지침 메시지
        parser_message = {"role": "system", "content": f"응답 형식 지침:\n{format_instructions}"}
        # 선택된 기능에 따른 추가 프롬프트 구성
        feature_mapping = {
            "customize": "요구사항 커스터마이즈: 사용자의 알레르기, 선호, 상황(인원수, 식이 제한 등)에 따라 재료와 조리법을 자동 조정해야 해.",
            "diet": "다이어트 플랜: 사용자의 건강 목표(체중 감량, 저탄수, 고단백 등)에 맞춘 식단과 레시피를 제안해야 해.",
            "event": "이벤트 메뉴: 파티, 명절 등 특정 이벤트에 어울리는 메뉴와 조리 가이드를 제안해야 해.",
            "difficulty": "난이도 조정: 요리 초보~전문가까지 사용자의 수준에 맞춰 레시피 난이도를 조절해야 해."
        }
        composite_messages = [system_message, parser_message]
        # 'general'은 추가 프롬프트 없이 기본 체질 프롬프트만 사용
        if request.feature and request.feature in feature_mapping:
            composite_messages.append({"role": "system", "content": feature_mapping[request.feature]})
        composite_messages += request.messages
        resp = llm.invoke(composite_messages)
        content = resp.content
        print("메시지 내용: ", content)

        # 레시피 감지 플래그
        recipe_detected = False
        # PydanticOutputParser로 content 파싱
        try:
            # PydanticOutputParser로 단일 Recipe 파싱
            recipe_obj = parser.parse(content)
            recipes_list = [recipe_obj.dict()]
            response_message = json.dumps(recipes_list, ensure_ascii=False)
            recipe_detected = True
        except Exception as e:
            print('파싱 에러:', e)
            # fallback: JSON array 혹은 객체 형태인지 검사
            try:
                parsed = json.loads(content)
                # 객체 하나
                if isinstance(parsed, dict) and 'title' in parsed and 'ingredients' in parsed:
                    recipes_list = [parsed]
                    response_message = json.dumps(recipes_list, ensure_ascii=False)
                    recipe_detected = True
                # 배열 형태
                elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) \
                     and 'title' in parsed[0] and 'ingredients' in parsed[0]:
                    recipes_list = parsed
                    response_message = content
                    recipe_detected = True
                else:
                    recipes_list = []
                    response_message = content
            except Exception:
                recipes_list = []
                response_message = content
        # MongoDB에 대화 및 레시피 저장
        doc = {
            "session_id": request.session_id,
            "messages": request.messages,
            "recipes": recipes_list,
            "created_at": datetime.utcnow()
        }
        collection.insert_one(doc)
        # 레시피 여부 플래그 설정 (JSON 파싱 감지 기준)
        return ChatResponse(message=response_message, is_recipe=recipe_detected)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entrypoint for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.settings.HOST, port=config.settings.PORT) 