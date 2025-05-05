import os
from typing import Dict
import pathlib
from utils.prompt_loader import load_prompt
from model.recipe_model import recipe_evaluate_llm
import json

def evaluate_qa(history: str) -> Dict:
    '''
    주어진 대화를 LLM으로 평가하고 JSON으로 결과를 반환합니다.
    '''
    qa_evaluateprompt_template = load_prompt("constitution_recipe_evaluate/constitution_recipe_question.json")
    # serialize history if it's a list of message dicts
    if isinstance(history, list):
        history_str = json.dumps(history, ensure_ascii=False)
    else:
        history_str = history
    prompt = qa_evaluateprompt_template.format(qa_list=history_str)
    response = recipe_evaluate_llm.invoke([{'role': 'user', 'content': prompt}])
    
  
    content = response.content
    print(content)
    # JSON 파싱
    try:
        result = json.loads(content)
        score = evaluate_metric(result)
    except Exception as e:
        raise ValueError(f'JSON 파싱 실패: {e} 응답 내용: {content}')
    return result, score


def evaluate_recipe(history: str, recipe: str) -> Dict:
    '''
    주어진 레시피를 LLM으로 평가하고 JSON으로 결과를 반환합니다.
    '''

    recipe_evaluate_template = load_prompt("constitution_recipe_evaluate/consitution_recipe_recipe.json")
    csv_path = pathlib.Path("data/정리된_섭생표_데이터.csv")
    constitution_table = csv_path.read_text(encoding="utf-8")

    # serialize history if it's a list of message dicts
    if isinstance(history, list):
        history_str = json.dumps(history, ensure_ascii=False)
    else:
        history_str = history
    prompt = recipe_evaluate_template.format(dialogue=history_str, recipe_json=recipe, constitution_table=constitution_table)
    response = recipe_evaluate_llm.invoke([{'role': 'user', 'content': prompt}])
    
  
    content = response.content
    print(content)
    # JSON 파싱
    try:
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
    