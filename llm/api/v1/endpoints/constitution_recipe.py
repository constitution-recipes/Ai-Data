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
from utils.evaluator.recipe_evaluator import evaluate_qa, evaluate_recipe
from langchain_core.messages import HumanMessage, AIMessage
from model.recipe_model import get_recipe_llm
from model.get_llm import get_llm

LANGSMITH_TRACING = config.settings.LANGSMITH_TRACING
LANGSMITH_API_KEY = config.settings.LANGSMITH_API_KEY
LANGSMITH_ENDPOINT = config.settings.LANGSMITH_ENDPOINT
LANGSMITH_PROJECT_NAME = config.settings.LANGSMITH_PROJECT_NAME

# 환경 변수 설정
os.environ['OPENAI_API_KEY'] = config.settings.OPENAI_API_KEY
openai.api_key = config.settings.OPENAI_API_KEY

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    feature: Optional[str] = None
    messages: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]

class ChatResponse(BaseModel):
    message: str
    is_recipe: bool = False
    error: Optional[str] = None
    qa_result: Optional[List[Dict[str, str]]] = None
    recipe_result: Optional[List[Dict[str, str]]] = None
    qa_score: Optional[float] = None
    recipe_score: Optional[float] = None
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

def request_to_input(request: ChatRequest):
    prompt_template = load_prompt("constitution_recipe/consitiution_recipe_base_prompt.json")
    format_instructions = parser.get_format_instructions()
    formatted = prompt_template.format(format_instructions=format_instructions)
    system_message = {"role": "system", "content": formatted}
    parser_message = {"role": "system", "content": f"응답 형식 지침:\n{format_instructions}"}
    composite_messages = [system_message, parser_message]
    for qa in request.messages:
        if qa['role'] == 'user':
            composite_messages.append(HumanMessage(content=qa['content']))
        else:
            composite_messages.append(AIMessage(content=qa['content']))
    return composite_messages
def output_to_json_response(request: ChatRequest,content: str):
    # 레시피 감지 플래그
    recipe_detected = False
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
    
    # 레시피 평가
    if recipe_detected:
        # 레시피 평가 프롬프트 로드
        qa_result, qa_score = evaluate_qa(request.messages)
        print(f"[{datetime.now()}] 레시피 평가 결과: {qa_result}, 점수: {qa_score}")
        recipe_result, recipe_score = evaluate_recipe(request.messages, response_message)
        print(f"[{datetime.now()}] 레시피 평가 결과: {recipe_result}, 점수: {recipe_score}")
        print(f"[{datetime.now()}] 체질 레시피 응답 완료: is_recipe={recipe_detected}")
        return ChatResponse(message=response_message, is_recipe=recipe_detected, qa_result=qa_result, recipe_result=recipe_result, qa_score=qa_score, recipe_score=recipe_score)
    return ChatResponse(message=response_message, is_recipe=recipe_detected)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"[{datetime.now()}] 체질 레시피 요청 시작: session_id={request.session_id}, feature={request.feature}")
    try:
        composite_messages = request_to_input(request)
        resp = get_recipe_llm("recipe_llm").invoke(composite_messages)
    
        content = resp.content
        print("메시지 내용: ", content)

        chat_json_response = output_to_json_response(request,content)
        return chat_json_response
        # PydanticOutputParser로 content 파싱
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

class TestRequest(BaseModel):
    qa_history_json: str
    provider: str
    model: str
    prompt_str: str

class TestResponse(BaseModel):
    prompt_str: str
    provider: str
    model: str
    qa_result: List[Dict[str, Any]]
    qa_score: float
    recipe_result: List[Dict[str, Any]]
    recipe_score: float
    average_score: float

@router.post("/test", response_model=TestResponse, summary="모델 및 프롬프트 테스트")
async def test_constitution_recipe(req: TestRequest):
    try:
        # parse QA history JSON
        history = json.loads(req.qa_history_json)
        # QA 평가
        qa_result, qa_score = evaluate_qa(history)
        # 모델 인스턴스 생성
        llm_instance = get_llm(req.provider, req.model)
        # 메시지 구성: system prompt + 대화 히스토리
        system_msg = {"role": "system", "content": req.prompt_str}
        composite_messages = [system_msg]
        for msg in history:
            if msg.get("role") == "user":
                composite_messages.append(HumanMessage(content=msg.get("content")))
            else:
                composite_messages.append(AIMessage(content=msg.get("content")))
        # 레시피 생성
        resp = llm_instance.invoke(composite_messages)
        content = resp.content
        # 레시피 평가
        recipe_result, recipe_score = evaluate_recipe(history, content)
        average_score = (qa_score + recipe_score) / 2
        return TestResponse(
            prompt_str=req.prompt_str,
            provider=req.provider,
            model=req.model,
            qa_result=qa_result,
            qa_score=qa_score,
            recipe_result=recipe_result,
            recipe_score=recipe_score,
            average_score=average_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
