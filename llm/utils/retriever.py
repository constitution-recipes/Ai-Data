import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 환경 변수에서 API 키 사용
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
# Chroma DB 디렉토리 설정
vectorstore = Chroma(
    collection_name="constitution",
    persist_directory=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
    embedding_function=embeddings
) 