from core.config import settings
from langchain_openai import ChatOpenAI

# 레시피 생성용 LLM 초기화
recipe_llm = ChatOpenAI(
    model_name=settings.RECIPE_MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY
) 