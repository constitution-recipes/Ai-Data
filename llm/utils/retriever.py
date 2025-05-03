import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 환경 변수에서 API 키 사용
embeddings = OpenAIEmbeddings(openai_api_key="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")
# Chroma DB 디렉토리 설정
vectorstore = Chroma(
    collection_name="constitution",
    persist_directory=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
    embedding_function=embeddings
) 