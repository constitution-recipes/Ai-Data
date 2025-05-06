import os
from typing import Dict, List
import pathlib
from utils.prompt_loader import load_prompt
from core.config import settings
from model.recipe_model import get_recipe_llm
from prompt.get_prompt import get_prompt
from core.config import settings

def evaluate_qa(history: List[Dict]) -> Dict:
    '''
    주어진 대화를 LLM으로 평가하고 JSON으로 결과를 반환합니다.
    '''
    qa_evaluateprompt_template = get_prompt(settings.RECIPE_EVALUATE_QA_PROMPT_NAME)
    prompt = qa_evaluateprompt_template.format(qa_list=history)
    recipe_evaluate_llm = get_recipe_llm(settings.RECIPE_EVALUATE_LLM_NAME)
    response = recipe_evaluate_llm.invoke([{'role': 'user', 'content': prompt}])
    
  
    content = response.content
    print(content)
    # JSON 파싱
    try:
        import json
        result = json.loads(content)
        score = evaluate_metric(result)
    except Exception as e:
        raise ValueError(f'JSON 파싱 실패: {e} 응답 내용: {content}')
    return result, score


def evaluate_recipe(history:List[Dict], recipe: str) -> Dict:
    '''
    주어진 레시피를 LLM으로 평가하고 JSON으로 결과를 반환합니다.
    '''

    recipe_evaluate_template = get_prompt(settings.RECIPE_EVALUATE_RECIPE_PROMPT_NAME)
    csv_path = pathlib.Path(settings.SEOUPSENG_CSV_PATH)
    constitution_table = csv_path.read_text(encoding="utf-8")

    prompt = recipe_evaluate_template.format(dialogue=history,recipe_json=recipe,constitution_table=constitution_table)
    recipe_evaluate_llm = get_recipe_llm(settings.RECIPE_EVALUATE_LLM_NAME)
    response = recipe_evaluate_llm.invoke([{'role': 'user', 'content': prompt}])
    
  
    content = response.content
    print(content)
    # JSON 파싱
    try:
        import json
        result = json.loads(content)
        score = evaluate_metric(result)
    except Exception as e:
        raise ValueError(f'JSON 파싱 실패: {e} 응답 내용: {content}')
    return result, score

def evaluate_metric(evaluate_json):
    count = 0
    for qa in evaluate_json:
        if qa['answer'] == '예':
            count += 1
    return count / len(evaluate_json)
    