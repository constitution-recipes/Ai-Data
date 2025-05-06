from utils.prompt_loader import load_prompt
from langchain import hub
from prompt.constitution_recipe.consitiution_recipe_route_system import route_system_prompt_template
def get_prompt(prompt_name: str):
    if prompt_name == "constitution_recipe_question":
        return load_prompt("constitution_recipe_evaluate/constitution_recipe_question.json")
    elif prompt_name == "consitution_recipe_recipe":
        return load_prompt("constitution_recipe_evaluate/consitution_recipe_recipe.json")
    elif prompt_name == "history_abstract":
        return load_prompt("constitution_recipe/history_abstract_prompt.json")
    elif prompt_name == "rewrite_for_web":
        return load_prompt("constitution_recipe/rewrite_for_web_prompt.json")
    elif prompt_name == "constitution_diagnose_answer":
        return load_prompt("consitituion_diagnose/constitution_diagnose_answer_prompt.json")
    elif prompt_name == "constitution_diagnose":
        return load_prompt("consitituion_diagnose/constitution_diagnose_prompt.json")
    elif prompt_name == "constitution_recipe_base":
        return load_prompt("constitution_recipe/consitiution_recipe_base_prompt.json")
    elif prompt_name == "constitution_recipe_evaluate_qa":
        return load_prompt("constitution_recipe_evaluate/constitution_recipe_question.json")
    elif prompt_name == "constitution_recipe_evaluate_recipe":
        return load_prompt("constitution_recipe_evaluate/consitution_recipe_recipe.json")
    elif prompt_name == "constitution_recipe_route_system":
        return route_system_prompt_template
    elif prompt_name == "constitution_recipe_base_generate":
        return load_prompt("constitution_recipe/constitution_recipe_base_generate_prompt.json")
    elif prompt_name == "constitution_recipe_base_ask":
        return load_prompt("constitution_recipe/consitiution_recipe_base_ask_prompt.json")
    elif prompt_name == "doc_relevance":
        return hub.pull("langchain-ai/rag-document-relevance")
    elif prompt_name == "constitution_recipe_base_generate_best":
        return load_prompt("constitution_recipe/consitiution_recipe_base_generate_best_prompt.json")

