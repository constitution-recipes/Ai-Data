from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import RetrievalQA
from utils.retriever import vectorstore
from utils.prompt_loader import load_prompt
from model.constitution_model import constitution_llm
import traceback
from enum import Enum
from core.config import settings
from prompt.get_prompt import get_prompt

router = APIRouter()

# --- 요청 및 응답 모델 정의
class DiagnoseRequest(BaseModel):
    answers: List[Dict[str, str]] = Field(default_factory=list)

class DiagnoseResponse(BaseModel):
    constitution: str
    reason: str
    confidence: float
    can_diagnose: bool
    next_question: Optional[str] = None

class ConstitutionEnum(str, Enum):
    geumyang = "금양"
    geumeum = "금음"
    suyangeum = "수양"
    sueum = "수음"
    mogyang = "목양"
    mokeum = "목음"
    toyang = "토양"
    toeum = "토음"

class DiagnosisModel(BaseModel):
    constitution: ConstitutionEnum = Field(..., alias="체질")
    reason: str = Field(..., alias="진단이유")
    confidence: float
    class Config:
        allow_population_by_field_name = True

parser = PydanticOutputParser(pydantic_object=DiagnosisModel)
format_instructions = parser.get_format_instructions()

# --- LLM 및 RAG 설정
llm = constitution_llm

retriever = vectorstore.as_retriever()


async def generate_question(answers: List[Dict[str, str]]) -> str:
    # 시스템 프롬프트 로드
    history_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
    prompt = get_prompt(settings.CONSTITUTION_DIAGNOSE_ANSWER_PROMPT_NAME)
    formatted_system = prompt.format(qa_list=history_text)
    print("QAHISTORY:", history_text)
    # 대화 메시지 구성: system + (AIMessage(question), HumanMessage(answer))*
    messages = [SystemMessage(content=formatted_system)]
    for qa in answers:
        messages.append(AIMessage(content=qa['question']))
        messages.append(HumanMessage(content=qa['answer']))
    # LLM에 메시지 전달하여 질문 생성
    resp = await llm.agenerate([messages])
    question = resp.generations[0][0].text.strip()
    print("Generated question:", question)
    return question

async def perform_diagnose(answers: List[Dict[str, str]]) -> DiagnosisModel:
    # 프롬프트 로드 및 포맷
    prompt = get_prompt(settings.CONSTITUTION_DIAGNOSE_PROMPT_NAME)
    history_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
    # format_instructions를 포함하여 시스템 메시지 구성
    formatted_system = prompt.format(qa_list=history_text, format_instructions=format_instructions)
    print("DIAGNOSIS HISTORY:", history_text)
    # 메시지 리스트 구성: System → (AIMessage, HumanMessage)*
    messages = [SystemMessage(content=formatted_system)]
    for qa in answers:
        messages.append(AIMessage(content=qa['question']))
        messages.append(HumanMessage(content=qa['answer']))
    # LLM 호출
    resp = await llm.agenerate([messages])
    content = resp.generations[0][0].text.strip()
    print("Diagnosis LLM output:", content)
    # 결과 파싱
    parsed: DiagnosisModel = parser.parse(content)
    return parsed

# --- FastAPI 라우터
@router.post("/", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    try:
        print("request:", request)
        answers = request.answers or []
        # 1) 초기 질문
        if not answers:
            question = await generate_question(answers)
            return DiagnoseResponse(
                constitution="", reason="", confidence=0.0, can_diagnose=False, next_question=question
            )
        # 2) 추가 질문 (최소 8개 질문)
        if len(answers) < 8:
            question = await generate_question(answers)
            return DiagnoseResponse(
                constitution="", reason="", confidence=0.0, can_diagnose=False, next_question=question
            )
        # 3) 진단 수행
        diag_result = await perform_diagnose(answers)
        # 4) 신뢰도 판단 및 추가 질문
        if len(answers) < 10 and diag_result.confidence < 0.85:
            question = await generate_question(answers)
            return DiagnoseResponse(
                constitution="", reason="", confidence=0.0, can_diagnose=False, next_question=question
            )
        # 5) 최종 결과
        return DiagnoseResponse(
            constitution=diag_result.constitution,
            reason=diag_result.reason,
            confidence=diag_result.confidence,
            can_diagnose=True,
            next_question=None
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"diagnose internal error: {str(e)}")
