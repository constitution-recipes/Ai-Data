from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from pathlib import Path
import json

import core.config as config
from model.recipe_model import get_recipe_llm
from langchain_core.messages import SystemMessage
from utils.prompt_loader import load_prompt

router = APIRouter()

class PromptRegenerationRequest(BaseModel):
    prompt_path: str = Field(..., description="prompt 폴더 내 JSON 파일 경로 (예: constitution_recipe/consitiution_recipe_base_generate_best_prompt.json)")
    edit_instructions: str = Field(..., description="프롬프트를 보완하기 위한 수정사항")
    input_variables: Dict[str, Any] = Field(..., description="원본 프롬프트의 input_variables 값")

class PromptRegenerationResponse(BaseModel):
    best_prompt: str = Field(..., description="재생성된 최적의 프롬프트 내용")

@router.post("/recipe_prompt_regeneration", response_model=PromptRegenerationResponse, summary="프롬프트 재생성 API")
async def regenerate_prompt(req: PromptRegenerationRequest):
    """
    프롬프트 재생성 파이프라인:
    1. JSON 프롬프트 파일(load_prompt) 로드
    2. 수정사항을 반영하여 프롬프트 보완
    3. 보완된 프롬프트(best_prompt) 반환
    """
    try:
        original_prompt = load_prompt(req.prompt_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="원본 프롬프트를 찾을 수 없습니다.")

    # input_variables 유지하면서 수정사항 반영
    system_content = (
        "다음 원본 프롬프트를 아래 수정사항에 따라 보완하세요.\n\n"
        f"=== 원본 프롬프트 ===\n{original_prompt.template}\n\n"
        f"수정사항: {req.edit_instructions}\n\n"
        f"※ input_variables({original_prompt.input_variables})는 변경하지 마세요."
    )

    llm = get_recipe_llm(config.settings.RECIPE_LLM_NAME)
    # LLM 직접 호출
    response = llm.invoke([SystemMessage(content=system_content)])

    # OpenAIResponse 또는 dict 형태의 응답에서 content 추출
    best_prompt = getattr(response, "content", None) or response.get("query", "")

    # 수정된 프롬프트를 JSON 파일에 저장
    prompt_file_path = Path(__file__).resolve().parents[3] / "prompt" / req.prompt_path
    try:
        text = prompt_file_path.read_text(encoding="utf-8")
        data = json.loads(text)
        key = next(iter(data))
        data[key]["template"] = best_prompt
        # input_variables는 변경하지 않음
        prompt_file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프롬프트 저장 실패: {e}")

    return PromptRegenerationResponse(best_prompt=best_prompt) 