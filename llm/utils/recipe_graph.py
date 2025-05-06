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
당신은 문장을 간결하게 재구성하는 전문가입니다. 사용자의 문장을 아래 규칙에 따라 **정해진 형식**으로 바꿔주세요.

변환 규칙:
1. 사용자가 말한 **음식 이름**을 문장 맨 앞에 작성하세요. 요리명이 띄워쓰기가 되어있을 경우 붙여씁니다.
2. 뒤에는 사용자가 언급한 **원하는 음식 재료**를 나열하세요.
3. 피하거나 알레르기가 있는 음식 재료는 `'재료명'제외` 형식으로 적어주세요.
4. 특정 맛을 피할 경우 `'특정맛'제외` 형식으로 적어주세요.
4. 전체 출력은 띄어쓰기로 구분된 한 줄의 텍스트입니다.

예시:
- 입력: '금양체질이라 딸기를 피하고 싶어요. 소고기를 이용한 음식을 먹고 싶어요.'
  출력: `소고기 딸기제외`
- 입력: '소고기랑 마늘을 피하고 싶은데 미역국을 먹고 싶어요.'
  출력: `미역국 소고기제외 마늘제외`
- 입력: '저는 얼큰한 돼지국밥을 먹고 싶은데 땅콩에는 알레르기가 있고 무를 싫어해요.'
  출력: `돼지국밥 땅콩제외 무제외`
- 입력: '소음체질인데 매운음식을 피했으면 좋겠어요. 닭고기를 좋아해요. 닭도리탕이 먹고 싶어요.'
  출력: `닭도리탕 닭고기 매운맛제외`
- 입력: '목음체질이고 콩을 싫어하는데 햄 볶음밥을 먹을래요.'
  출력: `햄 볶음밥 콩제외`

사용자 문장:
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
너는 전문 요리사이며, 사용자 질문에 기반한 요리를 만들 때 반드시 아래에 제공된 관련 레시피(context)만 참고해야 해.

### 너의 작업 목표:
1. 사용자 질문에서 **요리명**, **사용하고 싶은 재료**, **제외해야 할 재료**를 추출해.
   - 제외해야 할 재료는 `'재료명'제외` 형태로 제공됨.
2. 아래 제공된 레시피(context) 중에서 **요리명이 유사하거나**, **사용할 재료가 포함되고**, **제외 재료가 포함되지 않은** 레시피를 중심으로 골라.
3. 해당 레시피를 바탕으로, 질문에 부합하는 **새로운 레시피(요리명, 재료, 조리방법 포함)**를 만들어줘.
4. 만약 제외해야 할 재료가 context 레시피에 포함되어 있더라도, 해당 재료는 **최종 레시피에 포함하지 말 것.**

---

### 참고할 수 있는 관련 레시피 목록:
{context}

---

### 사용자 질문:
{query}

---

### 출력 형식 (예시):
제목: '요리 이름'
재료: '재료 목록'
조리순서:  
1. '조리 과정'
2. ...  

주의: 질문에 나온 제외 재료는 절대 포함하지 말고, 반드시 context에서 근거를 찾아 사용해야 해.
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