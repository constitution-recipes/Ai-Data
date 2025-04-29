from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="AI-Data LLM Service",
    description="Microservice for LLM chat using OpenAI or LangChain",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}, ...]

class ChatResponse(BaseModel):
    message: str

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", 
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        resp = llm.invoke(request.messages)
        content = resp.content
        return ChatResponse(message=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entrypoint for running with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567) 