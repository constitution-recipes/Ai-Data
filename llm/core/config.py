from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: str            # OpenAI API 키
    MONGODB_URI: str               # MongoDB 연결 URL
    MONGODB_DB_NAME: str           # MongoDB DB 이름
    DIAGNOSIS_MODEL_NAME: str      # 체질 진단에 사용할 LLM 모델 이름
    RECIPE_MODEL_NAME: str         # 레시피 생성에 사용할 LLM 모델 이름               # LLM 모델 이름
    API_PREFIX: str                # REST API 엔드포인트 경로
    HOST: str                      # FastAPI 호스트
    PORT: int                      # FastAPI 포트
    BACKEND_URL: str               # 백엔드 서비스 API URL
    LANGSMITH_TRACING: bool = False # Langsmith 트레이싱 사용 여부
    LANGSMITH_API_KEY: Optional[str] = None # Langsmith API 키
    LANGSMITH_ENDPOINT: Optional[str] = None # Langsmith 엔드포인트
    LANGSMITH_PROJECT_NAME: Optional[str] = None # Langsmith 프로젝트 이름

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# 환경 변수 로드
settings = Settings()

OPENAI_API_KEY = settings.OPENAI_API_KEY
MONGODB_URI = settings.MONGODB_URI
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
DIAGNOSIS_MODEL_NAME = settings.DIAGNOSIS_MODEL_NAME
RECIPE_MODEL_NAME = settings.RECIPE_MODEL_NAME
API_PREFIX = settings.API_PREFIX
HOST = settings.HOST
PORT = settings.PORT
BACKEND_URL = settings.BACKEND_URL
LANGSMITH_TRACING = settings.LANGSMITH_TRACING
LANGSMITH_API_KEY = settings.LANGSMITH_API_KEY
LANGSMITH_ENDPOINT = settings.LANGSMITH_ENDPOINT
LANGSMITH_PROJECT_NAME = settings.LANGSMITH_PROJECT_NAME
