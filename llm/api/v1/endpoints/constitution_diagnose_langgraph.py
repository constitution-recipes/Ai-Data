# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Dict, Optional, TypedDict
# from langchain.chains import RetrievalQA
# from utils.retriever import vectorstore
# from utils.prompt_loader import load_prompt
# from langgraph.graph import StateGraph, START, END
# from langchain.output_parsers import PydanticOutputParser
# from langchain.schema import SystemMessage
# from model.constitution_model import constitution_llm
# import traceback

# router = APIRouter()

# # --- 요청 및 응답 모델 정의
# class DiagnoseRequest(BaseModel):
#     answers: List[Dict[str, str]] = Field(default_factory=list)

# class DiagnoseResponse(BaseModel):
#     constitution: str
#     reason: str
#     confidence: float
#     can_diagnose: bool
#     next_question: Optional[str] = None

# class DiagnosisModel(BaseModel):
#     constitution: str = Field(..., alias="체질")
#     reason: str = Field(..., alias="진단이유")
#     confidence: float
#     class Config:
#         allow_population_by_field_name = True

# parser = PydanticOutputParser(pydantic_object=DiagnosisModel)

# # --- LLM 및 RAG 설정
# llm = constitution_llm
# retriever = vectorstore.as_retriever()

# # --- LangGraph 상태 정의
# class ConstitutionState(TypedDict):
#     user_answers: List[Dict[str, str]]
#     last_question: Optional[str]
#     last_answer: Optional[str]
#     constitution: str
#     reason: str
#     confidence: float
#     can_diagnose: bool

# # --- LangGraph 노드
# async def question_node(state: ConstitutionState):
#     history = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in state['user_answers']])
#     prompt = load_prompt("consitituion_diagnose/constitution_diagnose_answer_prompt.json")
#     formatted = prompt.format(qa_list=history)
#     print("QAHISTORY:", history)
#     print("PROMPT:", formatted)
#     resp = await llm.agenerate([[SystemMessage(content=formatted)]])
#     state['last_question'] = resp.generations[0][0].text.strip()
#     print("Generated question:", state['last_question'])
#     return state

# async def answer_node(state: ConstitutionState):
#     print(f"답변 시작: {state['last_question']}")
#     if 'last_question' in state and 'last_answer' in state:
#         state['user_answers'].append({
#             "question": state['last_question'],
#             "answer": state['last_answer']
#         })
#     return state

# async def diagnose_node(state: ConstitutionState):
#     print(f"diagnose_node 시작: {state['user_answers']}")
#     prompt = load_prompt("consitituion_diagnose/constitution_diagnose_prompt.json")
#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         question_prompt=prompt
#     )
#     user_input = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in state['user_answers']])
#     result = await rag_chain.acall({"query": user_input})
#     content = result.get("result", result.get("text", ""))
#     parsed: DiagnosisModel = parser.parse(content)
#     state['constitution'] = parsed.constitution
#     state['reason'] = parsed.reason
#     state['confidence'] = parsed.confidence
#     state['can_diagnose'] = True
#     return state

# async def check_confidence(state: ConstitutionState):
#     print(f"check_confidence 시작: {state['confidence']}")
#     return "ask" if state.get("confidence", 0) < 0.7 else "save"

# async def save_node(state: ConstitutionState):
#     print(f"save_node 시작: {state['constitution']}")
#     return state

# # --- 진단 진행 분기
# def evaluate_next_node(state: ConstitutionState):
#     print(f"evaluate_next_node 시작: {len(state['user_answers'])}")
#     if len(state["user_answers"]) < 8:
#         return "ask"
#     elif len(state["user_answers"]) <= 10:
#         return "diagnose"
#     return "diagnose"

# # --- LangGraph 정의
# graph = StateGraph(state_schema=ConstitutionState)
# graph.add_node("ask", question_node)
# graph.add_node("answer", answer_node)
# graph.add_node("evaluate", lambda s: s)
# graph.add_node("diagnose", diagnose_node)
# graph.add_node("check_confidence", check_confidence)
# graph.add_node("save", save_node)

# graph.add_edge(START, "ask")
# graph.add_edge("ask", "answer")
# graph.add_edge("answer", "evaluate")
# graph.add_conditional_edges("evaluate", evaluate_next_node, {"ask": "ask", "diagnose": "diagnose"})
# graph.add_edge("diagnose", "check_confidence")
# graph.add_conditional_edges("check_confidence", check_confidence, {"ask": "ask", "save": "save"})
# graph.add_edge("save", END)

# compiled_graph = graph.compile()

# # --- FastAPI 라우터
# @router.post("/", response_model=DiagnoseResponse)
# async def diagnose(request: DiagnoseRequest):
#     try:
#         # 최초 호출: 질문 목록이 비어있으면 첫 질문 생성
#         if not request.answers:
#             init_state: ConstitutionState = {
#                 "user_answers": [],
#                 "last_question": None,
#                 "last_answer": None,
#                 "constitution": "",
#                 "reason": "",
#                 "confidence": 0.0,
#                 "can_diagnose": False
#             }
#             next_state = await question_node(init_state)
#             return DiagnoseResponse(
#                 constitution="",
#                 reason="",
#                 confidence=0.0,
#                 can_diagnose=False,
#                 next_question=next_state["last_question"]
#             )

#         # 이어지는 호출: 사용자 답변 기반으로 다음 단계 결정
#         state: ConstitutionState = {
#             "user_answers": request.answers,
#             "last_question": None,
#             "last_answer": None,
#             "constitution": "",
#             "reason": "",
#             "confidence": 0.0,
#             "can_diagnose": False
#         }
#         # 분기 1: 추가 질문 필요 여부 결정
#         next_action = evaluate_next_node(state)
#         if next_action == "ask":
#             next_state = await question_node(state)
#             return DiagnoseResponse(
#                 constitution="",
#                 reason="",
#                 confidence=0.0,
#                 can_diagnose=False,
#                 next_question=next_state["last_question"]
#             )
#         # 분기 2: 진단 수행
#         diag_state = await diagnose_node(state)
#         # 분기 3: 신뢰도 검증 후 추가 질문 또는 완료
#         follow_action = await check_confidence(diag_state)
#         if follow_action == "ask":
#             follow_state = await question_node(diag_state)
#             return DiagnoseResponse(
#                 constitution="",
#                 reason="",
#                 confidence=0.0,
#                 can_diagnose=False,
#                 next_question=follow_state["last_question"]
#             )
#         # 최종 진단 결과 반환
#         return DiagnoseResponse(
#             constitution=diag_state["constitution"],
#             reason=diag_state["reason"],
#             confidence=diag_state["confidence"],
#             can_diagnose=True,
#             next_question=None
#         )
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"diagnose internal error: {str(e)}")
