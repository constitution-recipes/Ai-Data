import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from core.config import settings

# 환경 변수에서 API 키 사용
recipe_embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY,
)
# Chroma DB 디렉토리 설정
recipe_vector_store = Chroma(
    embedding_function=recipe_embedding,
    collection_name="recipe_vector_store",
    persist_directory="./chroma.sqlite3",
)

recipe_retriever = recipe_vector_store.as_retriever(search_kwargs={"k": 4})
reteriver = recipe_vector_store.as_retriever(search_kwargs={"k": 4})
vectorstore = Chroma(
    embedding_function=recipe_embedding,
    collection_name="recipe_vector_store",
    persist_directory="./chroma.sqlite3e",
)


