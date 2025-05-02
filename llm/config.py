from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str            # OpenAI API 키
    MONGODB_URI: str               # MongoDB 연결 URL
    MONGODB_DB_NAME: str           # MongoDB DB 이름
    MODEL_NAME: str # LLM 모델 이름
    API_PREFIX: str     # REST API 엔드포인트 경로
    HOST: str                      # FastAPI 호스트
    PORT: int                          # FastAPI 포트
    BACKEND_URL: str               # 백엔드 서비스 API URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 환경 변수 로드
settings = Settings() 

OPENAI_API_KEY = settings.OPENAI_API_KEY
MONGODB_URI = settings.MONGODB_URI
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
MODEL_NAME = settings.MODEL_NAME
API_PREFIX = settings.API_PREFIX
HOST = settings.HOST
PORT = settings.PORT
BACKEND_URL = settings.BACKEND_URL

