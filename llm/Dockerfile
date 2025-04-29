# LLM 마이크로서비스용 Dockerfile
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 코드 복사
COPY . /app

# 의존성 설치
RUN pip install --no-cache-dir fastapi uvicorn openai python-dotenv langchain_openai

# 포트 노출
EXPOSE 8000

# 서비스 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4567"] 