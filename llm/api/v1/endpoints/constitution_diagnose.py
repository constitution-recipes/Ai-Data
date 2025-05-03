from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import SystemMessage
from langchain.chains import RetrievalQA
from utils.retriever import vectorstore
from utils.prompt_loader import load_prompt
from model.constitution_model import constitution_llm
import traceback

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

class DiagnosisModel(BaseModel):
    constitution: str = Field(..., alias="체질")
    reason: str = Field(..., alias="진단이유")
    confidence: float
    class Config:
        allow_population_by_field_name = True

parser = PydanticOutputParser(pydantic_object=DiagnosisModel)

# --- LLM 및 RAG 설정
llm = constitution_llm
retriever = vectorstore.as_retriever()

async def generate_question(answers: List[Dict[str, str]]) -> str:
    history = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
    prompt = load_prompt("consitituion_diagnose/constitution_diagnose_answer_prompt.json")
    formatted = prompt.format(qa_list=history)
    print("QAHISTORY:", history)
    print("PROMPT:", formatted)
    resp = await llm.agenerate([[SystemMessage(content=formatted)]])
    question = resp.generations[0][0].text.strip()
    print("Generated question:", question)
    return question

async def perform_diagnose(answers: List[Dict[str, str]]) -> DiagnosisModel:
    prompt = load_prompt("consitituion_diagnose/constitution_diagnose_prompt.json")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        question_prompt=prompt
    )
    user_input = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in answers])
    result = await rag_chain.acall({"query": user_input})
    content = result.get("result", result.get("text", ""))
    print("Diagnosis LLM output:", content)
    parsed: DiagnosisModel = parser.parse(content)
    return parsed

# --- FastAPI 라우터
@router.post("/", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    try:
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
        if len(answers) < 10 and diag_result.confidence < 0.7:
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
