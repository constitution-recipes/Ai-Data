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

def get_recipe_llm(model_name: str):
    if model_name == "recipe_llm":
        return recipe_llm()
    elif model_name == "recipe_evaluate_llm":
        return recipe_evaluate_llm()
    elif model_name == "recipe_generate_llm_v2":
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
    query: str
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
    llm = get_llm("openai", settings.RECIPE_MODEL_NAME)

    graph_builder = StateGraph(RecipeAgentState)



    route_system_prompt_template = """
    당신은 한의학 전문가로서 사용자가 주어진 질문들들 목록에 충분한 답변을 했는지 판단하고 'recipe_gen' 과 'ask_llm' 으로 routing 을 해야합니다.

    주어진 질문 목록:
    - 사용자의 체질 
    - 알레르기
    - 건강 상태 
    - 식이 제한
    - 선호 음식 
    - 조리 도구
    - 인원수

    모든 질문에 사용자의 답변을 들은 후, 사용자가 답변을 충분하게 했는지 판단 기준은 다음과 같습니다:
    - 알레르기, 식이제한, 인원수의 경우 사용자가 필수로 답변해야 합니다. 답변이 충분하지 않다고 판단되면 다시 물어보세요.
    - 체질, 건강 상태, 원하는 음식, 조리 도구의 경우 사용자가 선택적으로 답변할 수 있는 항목입니다. 답변이 충분하지 않다고 판단되어도 다음 질문으로 넘어갈 수 있습니다.

    """

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", route_system_prompt_template),
        ("user", "{query}")
    ])


    def route_agent(state: RecipeAgentState) -> Literal["recipe_gen", "ask_llm"]:
        query = state["query"]
        structured_llm = llm.with_structured_output(Route)
        router_chain = router_prompt | structured_llm
        route = router_chain.invoke({"query": query})
        return {"query" :route.target}



    def ask_llm(state: RecipeAgentState):
        """
        사용자의 추가 정보를 얻기 위한 후속 질문 생성
        """
        query = state["query"]
        ask_prompt = load_prompt("constitution_recipe/recipe_ask_prompt.json")
        ask_abstract_chain = ask_prompt | llm | StrOutputParser()
        response = ask_abstract_chain.invoke({"query": query})
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
        
        return {"context": docs}



    def history_abstract(state: RecipeAgentState):
        query = state["query"]
        history_abstract_prompt = load_prompt("constitution_recipe/history_abstract_prompt.json")
        history_abstract_chain = history_abstract_prompt | llm | StrOutputParser()
        response = history_abstract_chain.invoke({"query": query})
        return {"query": response}


    def rewrite_query_for_web(state: RecipeAgentState):
        rewirte_for_web_prompt = load_prompt("constitution_recipe/rewrite_for_web_template.json")
        duck_web_search_tool = DuckDuckGoSearchRun()
        query = state["query"]
        rewrite_for_web_chain = rewirte_for_web_prompt | llm | StrOutputParser()
        response = rewrite_for_web_chain.invoke({"query": query})
        return {"query": response}

    def web_search(state: RecipeAgentState):
        duck_web_search_tool = DuckDuckGoSearchRun()
        query = state["query"]
        result = duck_web_search_tool.invoke(query)
        print("web search result: ")
        print(result)
        
        return {"context": result}


    def check_recipe_relevance(state: RecipeAgentState):
        doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")
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



    recipe_prompt = load_prompt("constitution_recipe/constitution_recipe_base_generate_prompt.json")

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
        recipe_rag_chain = recipe_prompt | llm | PydanticOutputParser(pydantic_object=Recipe)
        response = recipe_rag_chain.invoke({"query": query, "context": context})
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
        route_agent,
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
    graph_builder.add_edge('generate', END)


    graph = graph_builder.compile()
    return graph
