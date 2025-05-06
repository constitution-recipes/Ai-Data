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
from prompt.get_prompt import get_prompt
from utils.evaluator.recipe_evaluator import evaluate_qa, evaluate_recipe
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from model.get_llm import get_llm
from model.recipe_model import get_recipe_llm
from core.config import settings

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
    # 사용자 컨텍스트 필드 추가
    allergies: Optional[List[str]] = None
    constitution: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = None
    health_conditions: Optional[str] = None

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
print("format_instructions",parser.get_format_instructions())

def request_to_input(request: ChatRequest):
    # 사용자 컨텍스트 정보를 system 메시지로 추가
    user_context_prompt = get_prompt(settings.CONSTITUTION_RECIPE_USER_CONTEXT_PROMPT_NAME)
    # 마지막 사용자 메시지를 query로 사용
    last_user_message = ""
    if request.messages:
        for m in reversed(request.messages):
            if m.get('role') == 'user':
                last_user_message = m.get('content', "")
                break
    formatted_context = user_context_prompt.format(
        allergies=request.allergies or [],
        constitution=request.constitution or "",
        dietary_restrictions=request.dietary_restrictions or [],
        health_conditions=request.health_conditions or "",
        query=last_user_message
    )
    composite_messages = [SystemMessage(content=formatted_context)]
    # 기존 대화 메시지를 HumanMessage/AIMessage로 변환
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
        qa_result, qa_score = evaluate_qa(request.messages)
        recipe_result, recipe_score = evaluate_recipe(request.messages, response_message)
        return ChatResponse(message=response_message, is_recipe=recipe_detected, qa_result=qa_result, recipe_result=recipe_result, qa_score=qa_score, recipe_score=recipe_score)
    return ChatResponse(message=response_message, is_recipe=recipe_detected)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"[{datetime.now()}] 체질 레시피 요청 시작: session_id={request.session_id}, feature={request.feature}")
    try:
        history_message = request_to_input(request)
        if "graph" in settings.RECIPE_LLM_NAME:
            resp = get_recipe_llm(settings.RECIPE_LLM_NAME).invoke({"query": history_message})
            content = resp['query']
        else:
            prompt_template = get_prompt(settings.CONSTITUTION_RECIPE_BASE_PROMPT_NAME)
            format_instructions = parser.get_format_instructions()
            formatted = prompt_template.format(format_instructions=format_instructions)
            composite_messages = [SystemMessage(content=formatted), SystemMessage(content=f"응답 형식 지침:\n{format_instructions}")]
            composite_messages.extend(history_message)
            resp = get_recipe_llm(settings.RECIPE_LLM_NAME).invoke(composite_messages)
            content = resp.content

        chat_json_response = output_to_json_response(request,content)
        return chat_json_response
        
    # Exception 처리
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
        import traceback
        print('constitution_recipe.py 예외 발생:', str(e))
        print(traceback.format_exc())
        error_msg = f"알 수 없는 오류: {str(e)}"
        print(f"[{datetime.now()}] {error_msg}")
        return ChatResponse(
            message="레시피를 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            is_recipe=False,
            error=error_msg
        )

class TestConversation(BaseModel):
    # Conversation identifiers: either 'id' or 'sid'
    id: Optional[str] = None
    messages: List[Dict[str, str]]

class TestRequest(BaseModel):
    message_list: List[TestConversation]
    # AI 공급자, 모델, 시스템 프롬프트
    provider: str
    model: str
    prompt_str: str

class TestItem(BaseModel):
    id: str
    qa_result: List[Dict[str, Any]]
    qa_score: float
    recipe_result: List[Dict[str, Any]]  # 각 항목에 question, answer, reason 포함
    recipe_score: float
    average_score: float  # recipe_score만으로 할당
    recipe_json: dict = None  # 실제 레시피 객체
    input_tokens: Optional[int] = None  # 입력 토큰 수
    output_tokens: Optional[int] = None  # 출력 토큰 수
    cost: Optional[float] = None  # 비용 (USD)

class TestResponse(BaseModel):
    results: List[TestItem]
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    avg_cost_per_message: Optional[float] = None  # 메시지당 평균 비용 추가

# 모델별 가격 정보 (1M 토큰당 USD)
MODEL_PRICING = {
    # OpenAI 모델
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini-2024-07-18": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    
    # Gemini 모델
    "gemini-2.5-flash-preview-04-17": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro-preview-03-25": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-pro": {"input": 0.25, "output": 0.75},
    
    # Anthropic 모델
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-2": {"input": 8.00, "output": 24.00}
}

# 토큰 사용량 및 비용 계산 함수
def calculate_tokens_and_cost(provider: str, model: str, response):
    tokens = {"input": 0, "output": 0}
    
    # 프로바이더 및 모델별 토큰 정보 추출
    try:
        if provider == "openai":
            tokens["input"] = response.response_metadata["token_usage"]["prompt_tokens"]
            tokens["output"] = response.response_metadata["token_usage"]["completion_tokens"]
        elif provider == "gemini":
            tokens["input"] = response.usage_metadata["input_tokens"]
            tokens["output"] = response.usage_metadata["output_tokens"]
        elif provider == "claude":
            tokens["input"] = response.response_metadata["usage"]["input_tokens"]
            tokens["output"] = response.response_metadata["usage"]["output_tokens"]
    except Exception as e:
        print(f"토큰 정보 추출 오류: {e}")
        return 0, 0, 0.0
    
    # 비용 계산
    cost = 0.0
    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
        # 1M 토큰당 가격을 실제 토큰 수에 맞게 변환
        cost = (tokens["input"] * pricing["input"] / 1000000) + (tokens["output"] * pricing["output"] / 1000000)
    else:
        print(f"가격 정보가 없는 모델: {model}")
    
    return tokens["input"], tokens["output"], cost

@router.post("/test", response_model=TestResponse, summary="모델 및 프롬프트 테스트 (다중 대화 처리)")
async def test_constitution_recipe(req: TestRequest):
    print(f"[{datetime.now()}] 체질 레시피 테스트 요청 시작: message_list={req}")
    try:
        results: List[TestItem] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        message_count = len(req.message_list)
        
        for conv in req.message_list:
            history = conv.messages
            # QA 평가
            qa_result, qa_score = evaluate_qa(history)
            # 동적 LLM 인스턴스 생성
            llm_instance = get_llm(req.provider, req.model)
            # 시스템 프롬프트 + 대화 메시지 구성
            composite: List[Any] = [SystemMessage(content=req.prompt_str)]
            for msg in history:
                if msg.get("role") == "user":
                    composite.append(HumanMessage(content=msg["content"]))
                else:
                    composite.append(AIMessage(content=msg["content"]))
            # 레시피 생성
            resp = llm_instance.invoke(composite)
            
            # 토큰 사용량 및 비용 계산
            input_tokens, output_tokens, cost = calculate_tokens_and_cost(req.provider, req.model, resp)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            
            # 레시피 평가
            recipe_result, recipe_score = evaluate_recipe(history, resp.content)
            # 평균 점수: recipe_score만 사용
            average_score = recipe_score
            conv_id = conv.id or conv.sid or ""
            # PydanticOutputParser를 활용해 Recipe 객체 파싱
            try:
                recipe_obj = parser.parse(resp.content)
                recipe_json = recipe_obj.dict()
            except Exception as parse_err:
                print('test_constitution_recipe parser error:', parse_err)
                recipe_json = {}
            results.append(TestItem(
                id=conv_id,
                qa_result=qa_result,
                qa_score=qa_score,
                recipe_result=recipe_result,
                recipe_score=recipe_score,
                average_score=average_score,
                recipe_json=recipe_json,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            ))
            print("results", results)
        
        # 메시지 수가 0이면 오류 방지를 위해 1로 설정
        avg_cost_per_message = total_cost / max(message_count, 1)
        
        return TestResponse(
            results=results, 
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            avg_cost_per_message=avg_cost_per_message  # 메시지당 평균 비용 추가
        )
    except Exception as e:
        import traceback
        print('constitution_recipe.py 예외 발생:', str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    
