from utils.prompt_loader import load_prompt
def get_prompt(prompt_name: str):
    if prompt_name == "constitution_recipe_question":
        return load_prompt("constitution_recipe_evaluate/constitution_recipe_question.json")
    elif prompt_name == "consitution_recipe_recipe":
        return load_prompt("constitution_recipe_evaluate/consitution_recipe_recipe.json")
    elif prompt_name == "history_abstract":
        return load_prompt("constitution_recipe/history_abstract_prompt.json")
    elif prompt_name == "rewrite_for_web":
        return load_prompt("constitution_recipe/rewrite_for_web_prompt.json")
