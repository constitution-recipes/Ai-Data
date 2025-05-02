import os
from pathlib import Path
from langchain.prompts import PromptTemplate


def load_prompt(filename: str) -> PromptTemplate:
    """
    지정된 filename (.md) 프롬프트 파일을 읽어 PromptTemplate으로 반환합니다.
    파일 경로: <project_root>/Ai-Data/llm/utils/prompt/{filename}
    """
    # llm 디렉토리 내 prompt 폴더 참조
    base_path = Path(__file__).resolve().parent.parent / "prompt"
    file_path = base_path / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return PromptTemplate.from_template(text) 