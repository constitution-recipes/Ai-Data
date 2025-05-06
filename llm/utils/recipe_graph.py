from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langchain_community.tools import DuckDuckGoSearchRun # duckduckgosearchrun 은 무료!!!

from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever

from langgraph.graph import StateGraph, START, END

from langchain import hub

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal

from dotenv import load_dotenv



load_dotenv(r"./env", override=True)

# llm
mini_llm = ChatOpenAI(model="gpt-4.1-mini")
nano_llm = ChatOpenAI(model="gpt-4.1-nano")

# embedding_func
embedding_func = OpenAIEmbeddings(model="text-embedding-3-large")

# vector_store
def make_bm25_retriever(csv_file_path):
    df = pd.read_csv(csv_file_path)

    documents = []
    for _, row in df.iterrows():
        try:
            ingredients = eval(row["재료"])
            steps = eval(row["조리순서"])
        except Exception:
            ingredients = []
            steps = []

        content = (
            f"제목: {row['제목']}\n"
            f"재료: {', '.join(ingredients)}\n"
            f"조리순서: {', '.join(steps)}"
        )

        documents.append(Document(
            page_content=content,
            metadata={
                "name": row["제목"]
            }
        ))
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 5  # top-k 개수 설정
    return retriever

retriever = make_bm25_retriever("./recipe.csv")
# vector_store = Chroma(
#     embedding_function=embedding_func,
#     collection_name="recipe_vector_store",
#     persist_directory="./recipe_vector_store",
# )
# retriever = vector_store.as_retriever(search_kwargs={"k": 4})


class RecipeAgentState(TypedDict):
    query: str
    context: list
    answer: str


# retrieve
change_query_template = """
당신은 문장을 변환하는 전문가야. 주어진 문장을 아래 규칙에 변형해줘.

규칙:
주어진 문장에서 음식명과 음식재료를 나열하며 알레르기가 있거나 피해야 하는 음식은 재료명 뒤에 제외를 붙인다.

예시:
- '금양체질이라 딸기를 피하고 싶어요. 소고기를 이용한 음식을 먹고 싶어요' -> '소고기 딸기제외'
- '소고기랑 마늘을 피하고 싶은데 미역국을 먹고 싶어요' -> '미역국 소고기제외 마늘제외'

문장:
{query}
"""

change_query_prompt = PromptTemplate.from_template(change_query_template)


def change_query(state: RecipeAgentState):
    query = state["query"]
    print(f"input query: {query}")
    
    change_query_chain = change_query_prompt | mini_llm | StrOutputParser()
    response = change_query_chain.invoke({"query": query})
    print(f"changed query: {response}")
    return {"query": response}


def retrieve(state: RecipeAgentState):
    """
    사용자의 질문에 기반하여 벡터 스토어에서 문서 검색
    """
    query = state['query']
    docs = retriever.invoke(query)
    # return {"retrieve_context", docs}
    print("retrieve docs: ")
    print(docs)
    
    return {"context": docs}


doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")


def check_recipe_relevance(state: RecipeAgentState):
    """주어진 state 를 기반으로 문서의 관련성 판단"""
    query = state["query"]
    context = state["context"]
    # retrieve_context = state["retrieve_context"]
    # chain
    relevance_chain = doc_relevance_prompt | nano_llm
    response = relevance_chain.invoke({"question": query, "documents": context})
    
    if response["Score"] == 1:
        print("checked: relevent")
        return "relevant"

    print("checked: no relevent")
    return "no_relevant"


duck_web_search_tool = DuckDuckGoSearchRun()


def web_search(state: RecipeAgentState):
    query = state["query"]
    web_query = f"요리레시피 {query}"
    result = duck_web_search_tool.invoke(web_query)
    print("web search result: ")
    print(result)
    
    return {"context": result}
    # return {"web_search_context": result}


recipe_template = """
너는 전문 요리사로서 아래 관련 레시피를 참고해서 질문에 기반한 새로운 레시피를 만들어줘.
관련 레시피:
{context}

질문:
{query}
"""

recipe_prompt = PromptTemplate.from_template(recipe_template)

def generate(state: RecipeAgentState):
    query = state["query"]
    # retrieve_context = state["retrieve_context"]
    # web_search_context = state["web_search_context"] # 삭제가능
    context = state["context"]
    # chain
    """ 
    - 필요시 small_llm -> nano_llm
    *** output 에 따라 StrOutputParser() 수정 *** 
    
    """
    recipe_rag_chain = recipe_prompt | nano_llm | StrOutputParser() 
    response = recipe_rag_chain.invoke({"query": query, "context": context})
    return {"answer": response}

# graph builder
graph_builder = StateGraph(RecipeAgentState)

# add node
graph_builder.add_node("change_query", change_query)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node("web_search", web_search)

graph_builder.add_edge(START, 'change_query')
graph_builder.add_edge("change_query", "retrieve")

graph_builder.add_conditional_edges(
    "retrieve",
    check_recipe_relevance,
    {
        "relevant": "generate",
        "no_relevant": "web_search"
    }
)
graph_builder.add_edge("web_search", "generate")
graph_builder.add_edge('generate', END)

# add edge
graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    "retrieve",
    check_recipe_relevance,
    {
        "relevant": "generate",
        "no_relevant": END
    }
)
graph_builder.add_edge('generate', END)

recipe_graph = graph_builder.compile()