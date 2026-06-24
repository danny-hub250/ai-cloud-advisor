# ☁️ AI Public Cloud Advisor

Azure vs AWS 클라우드 추천 AI 어드바이저 — LangChain + LangGraph + RAG + FastAPI + Streamlit

---

## 개요

사용자의 서비스 요구사항(트래픽, 지역, 예산, 업종, 컴플라이언스 등)을 입력하면 **AWS 또는 Azure 중 최적의 클라우드**를 추천하고, 초기 아키텍처 및 월 비용 추정을 제공합니다.

단순한 LLM 호출이 아닌 **4단계 멀티에이전트 파이프라인**으로 동작합니다.

---

## 소스 구조

```
ai-cloud-advisor/
│
├── .env                    ← API 키, 모델명, 벡터DB 경로 등 환경변수
├── .env.example            ← 환경변수 템플릿
├── requirements.txt        ← 패키지 의존성
├── Dockerfile              ← 컨테이너 빌드 설정
├── app_streamlit.py        ← 웹 UI
│
├── api/
│   └── main.py             ← REST API 서버 (FastAPI)
│
├── agents/
│   ├── graph.py            ← 멀티에이전트 워크플로우 (LangGraph)
│   ├── prompts.py          ← LLM 프롬프트 템플릿
│   └── tools.py            ← 비용 계산 · RAG 검색 도구
│
├── rag/
│   ├── ingest.py           ← 문서를 벡터DB에 저장
│   ├── retriever.py        ← 벡터DB 유사 문서 검색
│   └── data/
│       ├── aws_basics.txt
│       └── azure_basics.txt
│
└── utils/
    └── model.py            ← LLM 클라이언트 초기화
```

---

## 데이터 흐름

```
사용자 입력 (Streamlit UI)
        ↓
  FastAPI POST /advise
        ↓
  LangGraph 4단계 파이프라인
  ┌──────────────────────────────────────────────────┐
  │  1. node_planner  → 요구사항 분석 & 계획 수립    │
  │         ↓                                        │
  │  2. node_research → RAG로 AWS/Azure 지식 검색    │
  │         ↓                                        │
  │  3. node_cost     → 월 비용 추정 계산            │
  │         ↓                                        │
  │  4. node_final    → 최종 추천 JSON 생성          │
  └──────────────────────────────────────────────────┘
        ↓
  결과 반환 → Streamlit에 표시
```

---

## 기술 스택

| 기술 | 역할 |
|---|---|
| **LangGraph** | 에이전트 워크플로우 순서 제어 |
| **LangChain** | 메시지 타입 표준화 |
| **Azure OpenAI / OpenAI** | LLM 추론 (계획·분석·추천) |
| **ChromaDB + MiniLM** | 지식 문서 저장 & 의미 검색 (RAG) |
| **FastAPI + uvicorn** | UI와 에이전트 사이 REST API |
| **Streamlit** | 사용자 인터페이스 |
| **Pydantic** | API 파라미터 검증 |
| **python-dotenv** | 환경변수·키 관리 |
| **Docker** | 배포용 컨테이너 패키징 |

### 각 기술 상세 설명

#### LangGraph
LangChain 위에 만들어진 **멀티에이전트 오케스트레이션 프레임워크**. 노드(실행 단계)와 엣지(연결 순서)로 에이전트 워크플로우를 그래프로 정의합니다. 단순 LLM 호출과 달리 **상태(state)** 를 단계별로 전달·공유할 수 있어 복잡한 AI 파이프라인 구성에 적합합니다.

#### LangChain
LLM 애플리케이션 개발을 위한 프레임워크. 이 프로젝트에서는 `HumanMessage`, `SystemMessage` 등 **메시지 타입 표준화** 용도로 사용합니다.

#### Azure OpenAI / OpenAI
실제 LLM 추론을 담당하는 API. `utils/model.py`에서 환경변수에 따라 Azure OpenAI 또는 일반 OpenAI 중 자동 선택합니다.

#### ChromaDB
**벡터 데이터베이스**. 텍스트를 수치 벡터로 변환해 저장하고, 의미적으로 유사한 문서를 빠르게 검색합니다. AWS·Azure 지식 문서를 저장해 RAG 검색에 활용합니다.

#### ONNX MiniLM (DefaultEmbeddingFunction)
텍스트를 벡터로 변환하는 **임베딩 모델**. `all-MiniLM-L6-v2`를 ONNX 형식으로 로컬 실행합니다. OpenAI API 호출 없이 로컬에서 임베딩을 생성합니다.

#### RAG (Retrieval-Augmented Generation)
LLM의 학습 데이터 한계를 극복하기 위해 외부 문서를 검색(`Retrieval`)한 뒤 LLM 프롬프트에 추가해 근거 기반 답변을 생성(`Generation`)하는 설계 패턴입니다.

#### FastAPI + uvicorn
Python에서 가장 빠른 **REST API 프레임워크**. Pydantic 모델로 요청/응답을 자동 검증하고, `/docs`에서 Swagger UI를 자동 생성합니다. uvicorn은 FastAPI를 실행하는 ASGI 웹 서버입니다.

#### Pydantic
**데이터 검증 라이브러리**. FastAPI의 Request/Response 모델 정의에 사용합니다. 타입 선언만으로 자동 검증하며 잘못된 값에 400 에러를 반환합니다.

#### Streamlit
**Python 코드만으로 웹 UI를 만드는 프레임워크**. HTML/CSS/JS 없이 대화형 웹앱을 빠르게 구성할 수 있습니다.

#### Docker
애플리케이션을 **컨테이너**로 패키징해 어떤 환경에서도 동일하게 실행할 수 있게 합니다.

---

## 설치 및 실행

### 사전 요구사항
- Python 3.11 이상
- OpenAI API 키 또는 Azure OpenAI 리소스

### 1. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 API 키를 입력합니다.

```env
# OpenAI 사용 시
OPENAI_API_KEY=sk-...

# Azure OpenAI 사용 시 (우선 적용)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=<your-key>

MODEL_NAME=gpt-4o-mini
VECTOR_DB_DIR=.chroma
```

### 2. 가상환경 및 패키지 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. RAG 데이터 인제스트 (최초 1회)

```bash
python rag/ingest.py
```

### 4. FastAPI 서버 실행 (터미널 1)

```bash
uvicorn api.main:app --port 8000 --reload
```

### 5. Streamlit UI 실행 (터미널 2)

```bash
streamlit run app_streamlit.py
```

브라우저에서 http://localhost:8501 접속

---

## API

### POST /advise

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `prompt` | string | 필수 | 요구사항 자유 입력 |
| `monthly_users` | int | 100000 | 월간 사용자 수 |
| `region` | string | apac | 주요 서비스 지역 (apac/us/eu) |
| `storage_gb` | int | 200 | 스토리지 용량 (GB) |
| `compute_tier` | string | medium | 컴퓨팅 규모 (small/medium/large) |
| `industry` | string | general | 업종 |
| `compliance` | string | null | 컴플라이언스 요구사항 |
| `existing_cloud` | string | null | 현재 사용 중인 클라우드 |

API 문서: http://localhost:8000/docs

---

## Docker로 실행

```bash
docker build -t ai-cloud-advisor .
docker run -p 8000:8000 --env-file .env ai-cloud-advisor
```

---

## 라이선스

MIT
