# 🔊 Ai-Data

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white"> <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white"> <img src="https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=LangChain&logoColor=white"> <img src="https://img.shields.io/badge/ChromaDB-40496D?style=for-the-badge&logo=ChromaDB&logoColor=white"> <img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=MongoDB&logoColor=white"> <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white"> <img src="https://img.shields.io/badge/docker--compose-2496ED?style=for-the-badge&logo=docker&logoColor=white">

## Project Overview
Ai-Data는 Constitution-Recipe 웹 애플리케이션의 AI 기반 데이터 파이프라인 및 LLM(대형 언어 모델) 모듈을 제공합니다. 주요 기능으로는:
- **크롤링**: `crawling/` 디렉토리의 스크립트를 통해 레시피 데이터를 수집 및 전처리
- **데이터 저장**: `data/` 폴더에 가공된 레시피 CSV 저장
- **LLM API**: FastAPI와 LangChain을 활용한 질의·응답 서비스 (`llm/`)
- **Docker 배포**: Dockerfile과 Docker Compose를 통한 컨테이너화 및 배포

## Project Structure
```
Ai-Data
 ┣ .gitignore
 ┣ crawling
 ┃ ┣ ttrecipe_crawl.py              # 단일 레시피 크롤러
 ┃ ┗ ttrecipe_crawl_multi.py        # 다중 페이지 크롤러
 ┣ data
 ┃ ┗ recipe_mainv2.csv              # 수집된 레시피 데이터
 ┣ llm
 ┃ ┣ Dockerfile                     # FastAPI 컨테이너 이미지 정의
 ┃ ┣ requirements.txt               # Python 패키지 목록
 ┃ ┣ main.py                        # FastAPI 앱 진입점
 ┃ ┣ api/                           # API 라우터 및 핸들러
 ┃ ┣ prompt/                        # 프롬프트 템플릿
 ┃ ┣ model/                         # 모델 인터페이스 및 로딩
 ┃ ┣ chroma_db/                     # ChromaDB 설정 및 초기화
 ┃ ┣ core/                          # 비즈니스 로직 핵심 모듈
 ┃ ┣ utils/                         # 유틸리티 함수
 ┃ ┗ evaluate/                      # 평가 스크립트
 ┗ README.md
```

## Usage
### 1. 데이터 크롤링
```bash
# 단일 크롤링
python crawling/ttrecipe_crawl.py
# 다중 페이지 크롤링
python crawling/ttrecipe_crawl_multi.py
```

### 2. LLM API 설정
```bash
cd llm
pip install -r requirements.txt
cp .env.example .env
# .env에 OPENAI_API_KEY 등 환경 변수 설정
```  

### 3. 로컬 실행 (FastAPI)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Docker 배포
```bash
cd llm
docker build -t llm .
docker run  -p 8000:8000 llm:latest
```

## License
This project is licensed under the MIT License.
