from core.config import settings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import StrOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from model.get_llm import get_llm
from langgraph.graph import START, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from utils.prompt_loader import load_prompt
from utils.retriever import recipe_retriever
from langchain import hub
from typing import TypedDict
from prompt.get_prompt import get_prompt
from utils.evaluator.recipe_evaluator import evaluate_recipe

def get_recipe_llm(model_name: str,):
    if model_name == "recipe_llm":
        return recipe_llm()
    elif model_name == "recipe_evaluate_llm":
        return recipe_evaluate_llm()
    elif model_name == "recipe_generate_llm_graph":
        return recipe_graph_llm()

def recipe_llm():
    return ChatOpenAI(
            model_name=settings.RECIPE_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY)

def recipe_evaluate_llm():
    return ChatOpenAI(
            model_name=settings.RECIPE_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY)

class RecipeAgentState(TypedDict):
    query: list[dict]
    context: list
    answer: str

### 레시피 진단 워크플로우
class Route(BaseModel):
    target: Literal["recipe_gen", "ask_llm"] = Field(
        description="query 에 대한 분류 target"
    )

class Recipe(BaseModel):
    title: str = Field(..., description="레시피 제목")
    description: str = Field(..., description="레시피 설명")
    difficulty: str = Field(..., description="난이도")
    cookTime: str = Field(..., description="조리 시간")
    ingredients: list[str] = Field(..., description="재료 목록")
    image: str = Field(..., description="이미지 URL")
    rating: float = Field(..., description="평점")
    suitableFor: str = Field(..., description="적합 대상 설명")
    reason: str = Field(..., description="레시피 생성 이유 설명")
    suitableBodyTypes: list[str] = Field(..., description="이 음식에 적합한 체질 리스트 (목양체질, 목음체질, 토양체질, 토음체질, 금양체질, 금음체질, 수양체질, 수음체질)")
    tags: list[str] = Field(..., description="태그 목록")
    steps: list[str] = Field(..., description="조리 단계 리스트")
    servings: str = Field(..., description="인분 정보")
    nutritionalInfo: str = Field(..., description="영양 정보")


def recipe_graph_llm():
    llm = get_llm(settings.RECIPE_MODEL_COMPANY_NAME, settings.RECIPE_MODEL_NAME)

    graph_builder = StateGraph(RecipeAgentState)

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt(settings.CONSTITUTION_RECIPE_ROUTE_SYSTEM_PROMPT_NAME)),
        ("user", "{query}")
    ])


    structured_llm = llm.with_structured_output(Route)
    def route_classify(state: RecipeAgentState) -> Literal["recipe_gen", "ask_llm"]:
        query = state["query"]
        print("route_classify 진입")
        router_chain = router_prompt | structured_llm
        route = router_chain.invoke({"query": query})
        print("route_classify 지나감")
        print("route target:", route.target)
        return route.target

    def route_agent(state: RecipeAgentState):
        # state update: only set query to classification result
        classification = route_classify(state)
        return {"query": classification}

    def ask_llm(state: RecipeAgentState):
        """
        사용자의 추가 정보를 얻기 위한 후속 질문 생성
        """
        query = state["query"]
        print("ask_llm 진입")
        ask_prompt = get_prompt(settings.CONSTITUTION_RECIPE_BASE_ASK_PROMPT_NAME)
        ask_abstract_chain = ask_prompt | llm | StrOutputParser()
        response = ask_abstract_chain.invoke({"query": query})
        print("ask_llm 지나감")
        return {"query": response}

    ### 레시피 생성 워크플로우

    def retrieve(state: RecipeAgentState):
        """
        사용자의 질문에 기반하여 벡터 스토어에서 문서 검색
        """
        query = state['query']
        docs = recipe_retriever.invoke(query)
        # return {"retrieve_context", docs}
        print("retrieve docs: ")
        print(docs)
        print("retrieve 지나감")
        
        return {"context": docs}



    def history_abstract(state: RecipeAgentState):
        query = state["query"]
        print("history_abstract 진입")
        history_abstract_prompt = get_prompt(settings.CONSTITUTION_RECIPE_HISTORY_ABSTRACT_PROMPT_NAME)
        history_abstract_chain = history_abstract_prompt | llm | StrOutputParser()
        response = history_abstract_chain.invoke({"query": query})
        print("history_abstract 지나감")
        return {"query": response}


    def rewrite_query_for_web(state: RecipeAgentState):
        rewirte_for_web_prompt = get_prompt(settings.CONSTITUTION_RECIPE_REWRITE_FOR_WEB_PROMPT_NAME)
        duck_web_search_tool = DuckDuckGoSearchRun()
        query = state["query"]
        rewrite_for_web_chain = rewirte_for_web_prompt | llm | StrOutputParser()
        response = rewrite_for_web_chain.invoke({"query": query})
        return {"query": response}

    def web_search(state: RecipeAgentState):
        duck_web_search_tool = DuckDuckGoSearchRun()
        query = state["query"]
        result = duck_web_search_tool.invoke(query)
        print("web_search 지나감")
        
        return {"context": result}


    def check_recipe_relevance(state: RecipeAgentState):
        doc_relevance_prompt = get_prompt(settings.CONSTITUTION_RECIPE_DOC_RELEVANCE_PROMPT_NAME)
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
        print("check_recipe_relevance 지나감")
        return "no_relevant"




    def generate(state: RecipeAgentState):
        recipe_prompt = get_prompt("constitution_recipe_base_generate_best")
        query = state["query"]
        # retrieve_context = state["retrieve_context"]
        # web_search_context = state["web_search_context"] # 삭제가능
        context = state["context"]
        # chain
        """ 
        - 필요시 small_llm -> nano_llm
        *** output 에 따라 StrOutputParser() 수정 *** 
        
        """
        recipe_rag_chain = recipe_prompt | llm | PydanticOutputParser(pydantic_object=Recipe)
        response = recipe_rag_chain.invoke({"query": query, "context": context})
        print("generate 지나감")
        return {"answer": response}


    graph_builder.add_node('retrieve', retrieve)
    graph_builder.add_node('history_abstract', history_abstract)
    graph_builder.add_node('generate', generate)
    graph_builder.add_node("web_search", web_search)
    graph_builder.add_node('ask_llm', ask_llm)
    graph_builder.add_node('route_agent', route_agent)

    graph_builder.add_edge(START, 'route_agent')
    graph_builder.add_conditional_edges(
        'route_agent',
        route_classify,
        {
            'ask_llm': 'ask_llm',
            'recipe_gen': 'history_abstract'
        }
    )
    graph_builder.add_edge('ask_llm',END)
    graph_builder.add_edge('history_abstract', 'retrieve')
    graph_builder.add_conditional_edges(
        "retrieve",
        check_recipe_relevance,
        {
            "relevant": "generate",
            "no_relevant": "web_search"
        }
    )
    graph_builder.add_edge("web_search", "generate")
    # generate 후 평가: score <= 0.8이면 재생성(retry), 그렇지 않으면 종료(accept)
    def evaluate_recipe_score(state: RecipeAgentState) -> Literal["retry", "accept"]:
        _, score = evaluate_recipe(state["query"], state["answer"])
        print("evaluate_recipe score:", score)
        return "retry" if score <= 0.8 else "accept"
    graph_builder.add_conditional_edges(
        "generate",
        evaluate_recipe_score,
        {
            "retry": "generate",
            "accept": END
        }
    )


    graph = graph_builder.compile()
    return graph
