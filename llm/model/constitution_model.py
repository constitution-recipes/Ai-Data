from core.config import settings
from langchain_openai import ChatOpenAI

# 체질 진단용 LLM 초기화
constitution_llm = ChatOpenAI(
    model_name=settings.DIAGNOSIS_MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY
) 