from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain.chains import RetrievalQA
from utils.retriever import vectorstore
from utils.prompt_loader import load_prompt
from langgraph.graph import Graph
from langchain.output_parsers import PydanticOutputParser
from model.constitution_model import constitution_llm

router = APIRouter()

# 요청/응답 모델
class DiagnoseRequest(BaseModel):
    answers: List[Dict[str, str]]  # [{'question': '...', 'answer': '...'}]

class DiagnoseResponse(BaseModel):
    constitution: str
    reason: str
    confidence: float
    can_diagnose: bool

# Pydantic model for parsing LLM JSON output
class DiagnosisModel(BaseModel):
    constitution: str = Field(..., alias="체질")
    reason: str = Field(..., alias="진단이유")
    confidence: float
    class Config:
        allow_population_by_field_name = True

# Output parser 초기화
parser = PydanticOutputParser(pydantic_object=DiagnosisModel)

# LLM 및 RAG 설정 (DB 접근 없음, vectorstore만 사용)
llm = constitution_llm
retriever = vectorstore.as_retriever()

# Graph 노드 정의
async def question_node(state):
    # LLM이 이전 Q&A를 바탕으로 다음 질문을 생성
    prompt = load_prompt("constitution/consitiuion_answer_prompt.json")
    # Q&A 히스토리 구성
    history = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in state['user_answers']])
    # PromptTemplate에 history 변수를 적용
    question_template = prompt.template.format(qa_list=history)
    resp = await llm.agenerate([[{"role": "system", "content": question_template}]])
    question = resp.generations[0][0].text.strip()
    # 사용자에게 물어볼 질문 저장
    state['last_question'] = question
    return state

async def answer_node(state):
    # 사용자가 답변한 내용을 state에 추가
    if 'last_answer' in state:
        state['user_answers'].append({'question': state.get('last_question', ''), 'answer': state['last_answer']})
    return state

async def diagnose_node(state):
    # RAG 기반 최종 진단 수행
    prompt = load_prompt("prompt/constitution/constitution_diagnose_prompt.json")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        question_prompt=prompt,
    )
    user_input = "\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}" for qa in state['user_answers']
    ])
    result = await rag_chain.acall({"query": user_input})
    # raw text 추출
    content = result.get("result", result.get("text", ""))
    # PydanticOutputParser로 JSON 파싱
    parsed: DiagnosisModel = parser.parse(content)
    state['constitution'] = parsed.constitution
    state['reason'] = parsed.reason
    state['confidence'] = parsed.confidence
    state['can_diagnose'] = True
    return state

async def check_confidence(state):
    # confidence 기준 이하 시 재질문, 이상 시 save 노드로 이동
    return 'ask' if state.get('confidence', 0) < 0.7 else 'save'

async def save_node(state):
    # 최종 결과 저장/종료 노드
    return state

# LangGraph 흐름 구성
graph = Graph()
graph.add_node('ask', question_node)
graph.add_node('answer', answer_node)
graph.add_node('evaluate', lambda s: s)  # placeholder for evaluation 노드
graph.add_node('diagnose', diagnose_node)
graph.add_node('check_confidence', check_confidence)
graph.add_node('save', save_node)
graph.add_edge('ask', 'answer')
graph.add_edge('answer', 'evaluate')
graph.add_edge('evaluate', lambda s: 'ask' if len(s.get('user_answers', [])) < 8 else ('diagnose' if len(s.get('user_answers', [])) <= 10 else 'diagnose'))
graph.add_edge('diagnose', 'check_confidence')
# check_confidence의 출력에 따라 ask 또는 save로 분기 처리
graph.add_conditional_edges(
    'check_confidence',
    check_confidence,
    {'ask': 'ask', 'save': 'save'}
)

@router.post("/", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    try:
        # 초기 상태 구성
        state = {
            'user_answers': request.answers,
            'constitution': '',
            'reason': '',
            'confidence': 0.0,
            'can_diagnose': False
        }
        final_state = await graph.run(state)
        return DiagnoseResponse(
            constitution=final_state.get('constitution', ''),
            reason=final_state.get('reason', ''),
            confidence=final_state.get('confidence', 0.0),
            can_diagnose=final_state.get('can_diagnose', False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
