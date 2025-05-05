from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langchain_community.tools import DuckDuckGoSearchRun # duckduckgosearchrun 은 무료!!!

from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END

from langchain import hub

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal

from dotenv import load_dotenv



load_dotenv(r"./env", override=True)

# llm
llm = ChatOpenAI(model="gpt-4.1-nano")

# embedding_func
embedding_func = OpenAIEmbeddings(model="text-embedding-3-large")

# vector_store
vector_store = Chroma(
    embedding_function=embedding_func,
    collection_name="recipe_vector_store",
    persist_directory="./recipe_vector_store",
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


class RecipeAgentState(TypedDict):
    query: str
    context: list
    answer: str


# retrieve
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
    relevance_chain = doc_relevance_prompt | llm
    response = relevance_chain.invoke({"question": query, "documents": context})
    
    if response["Score"] == 1:
        print("checked: relevent")
        return "relevant"

    print("checked: no relevent")
    return "no_relevant"



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
    recipe_rag_chain = recipe_prompt | llm | StrOutputParser() 
    response = recipe_rag_chain.invoke({"query": query, "context": context})
    return {"answer": response}

# graph builder
graph_builder = StateGraph(RecipeAgentState)

# add node
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node

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