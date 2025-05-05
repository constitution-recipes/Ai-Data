from fastapi import FastAPI, HTTPException, APIRouter
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import openai
import json
import traceback
from datetime import datetime
import core.config as config
import os
from utils.prompt_loader import load_prompt
from langsmith import traceable
from model.recipe_model import recipe_llm

LANGSMITH_TRACING = config.settings.LANGSMITH_TRACING
LANGSMITH_API_KEY = config.settings.LANGSMITH_API_KEY
LANGSMITH_ENDPOINT = config.settings.LANGSMITH_ENDPOINT
LANGSMITH_PROJECT_NAME = config.settings.LANGSMITH_PROJECT_NAME

# 환경 변수 설정
os.environ['OPENAI_API_KEY'] = config.settings.OPENAI_API_KEY
openai.api_key = config.settings.OPENAI_API_KEY

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    feature: Optional[str] = None
    messages: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]

class ChatResponse(BaseModel):
    message: str
    is_recipe: bool = False
    error: Optional[str] = None

# ChatOpenAI에 openai_api_key 파라미터로 전달하여 OpenAI 클라이언트에 API 키 설정
llm = recipe_llm

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


router = APIRouter()
@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"[{datetime.now()}] 체질 레시피 요청 시작: session_id={request.session_id}, feature={request.feature}")
    try:
        # 기본 system prompt를 외부 JSON 파일로 로드
        prompt_template = load_prompt("constitution_recipe/consitiution_recipe_base_prompt.json")
        formatted = prompt_template.format(format_instructions=format_instructions)
        system_message = {"role": "system", "content": formatted}
        # output parser 지침 메시지
        # 선택된 기능에 따른 추가 프롬프트 구성
        feature_mapping = {
            "customize": "요구사항 커스터마이즈: 사용자의 알레르기, 선호, 상황(인원수, 식이 제한 등)에 따라 재료와 조리법을 자동 조정해야 해.",
            "diet": "다이어트 플랜: 사용자의 건강 목표(체중 감량, 저탄수, 고단백 등)에 맞춘 식단과 레시피를 제안해야 해.",
            "event": "이벤트 메뉴: 파티, 명절 등 특정 이벤트에 어울리는 메뉴와 조리 가이드를 제안해야 해.",
            "difficulty": "난이도 조정: 요리 초보~전문가까지 사용자의 수준에 맞춰 레시피 난이도를 조절해야 해."
        }
        composite_messages = [system_message]
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
            print(f'[{datetime.now()}] 레시피 파싱 에러: {str(e)}')
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
            except Exception as json_err:
                print(f'[{datetime.now()}] JSON 파싱 에러: {str(json_err)}')
                recipes_list = []
                response_message = content
        
        print(f"[{datetime.now()}] 체질 레시피 응답 완료: is_recipe={recipe_detected}")
        return ChatResponse(message=response_message, is_recipe=recipe_detected)
    except openai.APIError as e:
        error_msg = f"OpenAI API 오류: {str(e)}"
        print(f"[{datetime.now()}] {error_msg}")
        traceback.print_exc()
        return ChatResponse(
            message="레시피 생성 중 AI 서비스 연결 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 
            is_recipe=False,
            error=error_msg
        )
    except openai.APIConnectionError as e:
        error_msg = f"OpenAI API 연결 오류: {str(e)}"
        print(f"[{datetime.now()}] {error_msg}")
        traceback.print_exc()
        return ChatResponse(
            message="AI 서비스 연결에 실패했습니다. 네트워크 연결을 확인하고 다시 시도해주세요.",
            is_recipe=False,
            error=error_msg
        )
    except openai.RateLimitError as e:
        error_msg = f"OpenAI API 속도 제한 오류: {str(e)}"
        print(f"[{datetime.now()}] {error_msg}")
        traceback.print_exc()
        return ChatResponse(
            message="현재 서비스가 많이 사용되고 있습니다. 잠시 후 다시 시도해주세요.",
            is_recipe=False,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"알 수 없는 오류: {str(e)}"
        print(f"[{datetime.now()}] {error_msg}")
        traceback.print_exc()
        return ChatResponse(
            message="레시피를 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            is_recipe=False,
            error=error_msg
        )
