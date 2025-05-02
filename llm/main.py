from fastapi import FastAPI
from api.v1.routers import api_router
import core.config as config

app = FastAPI(
    title="LLM Microservice",
    description="Microservice for LLM and RAG based constitution diagnosis and chat",
    version="1.0.0",
)

# API v1 라우터 등록
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.settings.HOST, port=config.settings.PORT) 