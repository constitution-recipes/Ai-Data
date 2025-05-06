from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from core.config import settings

def get_llm(provider: str, model_name: str):
    """provider와 model_name에 따라 LLM 인스턴스를 반환합니다."""
    # 현재는 OpenAI만 지원, 추후 Gemini/Claude 지원 가능
    if provider == 'openai':
        return ChatOpenAI(model_name=model_name, openai_api_key=settings.OPENAI_API_KEY)
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(model_name=model_name, google_api_key=settings.GEMINI_API_KEY)
    elif provider == 'claude':
        return ChatAnthropic(model_name=model_name, anthropic_api_key=settings.CLAUDE_API_KEY)