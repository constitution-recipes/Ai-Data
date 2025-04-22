import re
import time
import requests
import random
from bs4 import BeautifulSoup

import pandas as pd

save_review_fname = './data/recipe_review.csv'
save_recipe_fname = './data/recipe_main.csv'

type_dict = {
    '밥/죽/떡': '52',
    '면/만두': '53',
    '국/탕': '54',
    '찌개': '55',
    '메인반찬': '56',
    '양념/잼/소스': '58',
    '차/음료/술': '59',
    '디저트': '60',
    '퓨전': '61',
    '밑반찬': '63',
    '샐러드': '64',
    '양식': '65',
    '빵': '66',
    '스프': '68',
    '과자': '69',
} # cat4
situation_dict = {
    '일상': '12',
    '손님접대': '13',
    '도시락': '15',
    '간식': '17', 
    '초스피드': '18', 
    '술안주': '19', 
    '다이어트': '21', 
    '영양식': '43',
    '명절': '44', 
    '야식': '45'
}  # cat2
ingredient_dict = {
    '육류': '23',
    '해물류': '24',
    '건어물류': '25',
    '곡류': '26',
    '채소류': '28', 
    '버섯류': '31',
    '밀가루': '32',
    '쌀': '47', 
    '과일류': '48',
    '소고기': '70', 
    '돼지고기': '71', 
    '닭고기': '72', 
    '달걀/유제품': '50'}  # cat3
method_dict = {
    '끓이기': '1',
    '볶음': '6',
    '부침': '7',
    '찜': '8',
    '튀김': '9',
    '절임': '10',
    '조림': '36',
    '회': '37',
    '삶기': '38',
    '무침': '41',
    '비빔': '42',
    '굽기': '67'
} # cat1
sleep_time = 1
headers_text ={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Referer': 'https://www.google.com/',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1'}
recipe_idx = 1
list4df = []
list4rdf = []

def get_with_retry(url, headers=None, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

            # 여기서 "레시피 정보가 없습니다." 문구가 있는지 확인
            if "레시피 정보가 없습니다." in response.text:
                print("팝업: 레시피 정보가 없습니다. 반환값 None")
                return None

            return response
        except Exception as e:
            print(f"{attempt}번째 시도 실패: {e}")
            if attempt == max_retries:
                print("최대 재시도 횟수 도달. 종료합니다.")
                return None
            sleep_time = random.uniform(1, 3)
            print(f"{sleep_time:.1f}초 후 재시도합니다...")
            time.sleep(sleep_time)
    return None

try:
    session = requests.Session()
    session.headers.update(headers_text)
    for type_key, type_value in type_dict.items():
        for situ_key, situ_value in situation_dict.items():
            for ing_key, ing_value in ingredient_dict.items():
                for method_key, method_value in method_dict.items():
                    main_url = 'https://www.10000recipe.com/recipe/list.html?q=&query=&' \
                            'cat1={m}&cat2={s}&cat3={i}&cat4={t}' \
                            '&fct=&order=reco&lastcate=cat4&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource='\
                            .format(m=method_value, s=situ_value, i=ing_value, t=type_value)
                    response = get_with_retry(main_url, headers=headers_text)
                    if response:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        try:
                            page_len = int(soup.select_one('#contents_area_full > ul > div > b').get_text(strip=True)) // 40 + 1
                        except:
                            continue
                        print("페이지 갯수: ", page_len, '\n', "페이지 번호: ", main_url)
                        for page in range(1, page_len+1):
                            if page != 1:
                                main_url = main_url + '&page=' + str(page)
                                response = get_with_retry(main_url, headers=headers_text)
                                soup = BeautifulSoup(response.text, 'html.parser')
                                time.sleep(sleep_time*2)
                            if response:
                                sources = soup.select('#contents_area_full > ul > ul > li > div.common_sp_thumb > a')
                                for source in sources:
                                    recipe_url = 'https://www.10000recipe.com' + str(source).split('href')[1].split('"')[1]
                                    response_r = get_with_retry(recipe_url, headers=headers_text)
                                    soup_r = BeautifulSoup(response_r.text, 'html.parser')
                                    time.sleep(sleep_time)
                                    title = soup_r.select('#contents_area_full > div.view2_summary.st3 > h3')[0].text
                                    views = soup_r.select('#contents_area_full > div.view2_pic > div.view_cate.st2 > div > span')[0].text
                                    chef = soup_r.select('#contents_area_full > div.view2_pic > div.user_info2 > span')[0].text.strip()

                                    try:
                                        servings = soup_r.select('#contents_area_full > div.view2_summary.st3 > div.view2_summary_info > span.view2_summary_info1')[0].text
                                        servings = re.sub(r'[^0-9]', '', servings)

                                        cooking_time = soup_r.select('#contents_area_full > div.view2_summary.st3 > div.view2_summary_info > span.view2_summary_info2')[0].text
                                        difficulty = soup_r.select('#contents_area_full > div.view2_summary.st3 > div.view2_summary_info > span.view2_summary_info3')[0].text
                                    except:  # 작성이 안되어있는 경우
                                        servings = None
                                        cooking_time = None
                                        difficulty = None

                                    try:
                                        ingredient_s = soup_r.select('#divConfirmedMaterialArea > ul')
                                        ingredient = []
                                        for ing in ingredient_s:
                                            for i in ing.find_all('li'):
                                                ingredient.append(f"{i.get_text().split('\n')[3].strip()} {i.get_text().split('\n')[5].strip()}")
                                    except:
                                        continue

                                    try:
                                        intro = soup_r.select('#recipeIntro')[0].text.strip()
                                    except:
                                        intro = None

                                    try:
                                        cooking_order = []
                                        process = 1
                                        while True:
                                            try:
                                                cooking_order_s = soup_r.select('#stepdescr'+str(process))[0].text.strip()
                                                cooking_order.append(f"{process}. {cooking_order_s}")
                                                process += 1
                                            except:
                                                break
                                    except:
                                        continue

                                    try:
                                        hashtag_s = soup_r.select('#contents_area_full > div.view_step > div.view_tag')[0].find_all('a')
                                        hashtag = []
                                        for ht in hashtag_s:
                                            hashtag.append(ht.text[1:])  # 1: 샵 제거
                                    except:
                                        hashtag = None

                                    list4df.append([recipe_idx, type_key, situ_key, ing_key, method_key, title, recipe_url, views, chef, servings, cooking_time, difficulty, ingredient, intro, cooking_order, hashtag])
                                    recipe_idx += 1

                                    review_s = soup_r.select('#contents_area_full > div.view_reply > div > div.media')
                                    if len(review_s) == 0:
                                        continue
                                    for rev in review_s:
                                        review = []
                                        review.append(recipe_idx-1)
                                        review.append(rev.find('b').text.strip())  # 닉네임
                                        review.append(rev.find('h4').text.strip().split(' ')[-2])  # 날짜
                                        review.append(rev.find('h4').text.strip().split(' ')[-1])  # 시간

                                        star = str(rev.find('span')).count('icon_star2_on')
                                        review.append(star)  # 별점

                                        context = rev.find('p').text.strip()
                                        review.append(context)  # 내용
                                        list4rdf.append(review)

                                    add_review_s = soup_r.select('#moreViewReviewList > div.media')
                                    if len(add_review_s) == 0:
                                        continue
                                    for rev in add_review_s:
                                        review = []
                                        review.append(recipe_idx-1)
                                        review.append(rev.find('b').text.strip())  # 닉네임
                                        review.append(rev.find('h4').text.strip().split(' ')[-2])  # 날짜
                                        review.append(rev.find('h4').text.strip().split(' ')[-1])  # 시간

                                        star = str(rev.find('span')).count('icon_star2_on')
                                        review.append(star)  # 별점

                                        context = rev.find('p').text.strip()
                                        review.append(context)  # 내용
                                        list4rdf.append(review)

                                    if recipe_idx % 100 == 0:
                                        recipe_df = pd.DataFrame(list4df,
                                                                columns=['index', '종류별', '상황별', '재료별', '방법별', '제목', 'url', '조회수', '셰프', '인분',
                                                                        '조리시간', '난이도', '재료', '인트로', '조리순서', '해시태그'])
                                        review_df = pd.DataFrame(list4rdf, columns=['index', '닉네임', '작성날짜', '작성시간', '별점', '내용'])

                                        recipe_df.to_csv(save_recipe_fname, encoding='utf-8')
                                        review_df.to_csv(save_review_fname, encoding='utf-8')
                                        del [[recipe_df, review_df]]

                                        print(type_key, situ_key, ing_key, method_key)
                                        print(recipe_idx, "번째까지 완료")
                                        print('페이지: ', page, '제목: ', title)
                                        print()
                    else:
                        print('정상 연결되지 않았습니다: ', response.status_code)
except Exception as e:
    print(e)
    print(recipe_url)
    exit()

recipe_df = pd.DataFrame(list4df, columns=['index', '종류별', '상황별', '재료별', '방법별', '제목', 'url', '조회수', '셰프', '인분', '조리시간', '난이도', '재료', '인트로', '조리순서', '해시태그'])
review_df = pd.DataFrame(list4rdf, columns=['index', '닉네임', '작성날짜', '작성시간', '별점', '내용'])

pd.set_option('display.max_columns', None)
print(recipe_df)
print(review_df)

recipe_df.to_csv(save_recipe_fname, encoding='utf-8')
review_df.to_csv(save_review_fname, encoding='utf-8')