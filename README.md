☁️ AI Public Cloud Advisor

Azure vs AWS 클라우드 추천 AI 어드바이저 — LangChain + LangGraph + RAG + FastAPI + Streamlit

---

## 개요

사용자의 서비스 요구사항(트래픽, 지역, 예산, 업종, 컴플라이언스 등)을 입력하면 **AWS 또는 Azure 중 최적의 클라우드**를 추천하고, 초기 아키텍처 및 월 비용 추정을 제공합니다.

단순한 LLM 호출이 아닌 **4단계 멀티에이전트 파이프라인**으로 동작합니다.

---

<img width="650" height="650" alt="image" src="https://github.com/user-attachments/assets/ea3ee86b-3dab-45da-86a7-aa0682cf0330" />


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

---

### LangGraph

LangChain 위에 만들어진 **멀티에이전트 오케스트레이션 프레임워크**입니다. 단순한 LLM 호출과 달리, 노드(실행 단계)와 엣지(연결 순서)로 에이전트 워크플로우를 **그래프 구조**로 정의합니다.

**핵심 개념**

- **State**: 모든 노드가 공유하는 딕셔너리. 한 노드가 채운 값을 다음 노드가 읽음
- **Node**: state를 받아 처리 후 반환하는 Python 함수
- **Edge**: 노드 간 실행 순서 정의 (고정 순서 또는 조건 분기)

**이 프로젝트의 그래프 구조**

```
START → [plan] → [research] → [cost] → [final] → END
          ↓           ↓          ↓         ↓
       계획수립    RAG검색    비용계산   최종추천
```

**사용 위치**: `agents/graph.py`

```python
from langgraph.graph import StateGraph, END

sg = StateGraph(dict)
sg.add_node("plan",     node_planner)   # 1단계: 요구사항 분석
sg.add_node("research", node_research)  # 2단계: RAG 문서 검색
sg.add_node("cost",     node_cost)      # 3단계: 비용 계산
sg.add_node("final",    node_final)     # 4단계: 최종 추천 생성
sg.set_entry_point("plan")
sg.add_edge("plan", "research")
sg.add_edge("research", "cost")
sg.add_edge("cost", "final")
sg.add_edge("final", END)
graph = sg.compile()
```

---

### LangChain

LLM 애플리케이션 개발을 위한 **종합 프레임워크**입니다. 이 프로젝트에서는 LangGraph가 표준으로 사용하는 **메시지 타입 표준화** 용도로만 활용합니다. 실제 LLM 호출은 OpenAI SDK로 직접 수행합니다.

**패키지 구조**

| 패키지 | 역할 |
|---|---|
| `langchain-core` | 핵심 메시지 타입 (이 프로젝트에서 사용) |
| `langchain` | 체인·에이전트 등 고수준 기능 |
| `langgraph` | 그래프 기반 멀티에이전트 (별도 패키지) |

**사용 위치**: `agents/graph.py`

```python
from langchain_core.messages import HumanMessage, SystemMessage

# SystemMessage: LLM에게 역할 부여 (role: "system")
# HumanMessage: 사용자 질문 전달 (role: "user")

msgs = [
    SystemMessage(content="당신은 CloudAdvisor입니다..."),
    HumanMessage(content=f"{user_input}\n{COT_HINT}")
]
# llm_chat()에서 OpenAI 딕셔너리 형식으로 자동 변환
```

**Few-shot + Chain-of-Thought 적용**

```python
# agents/prompts.py
PLANNER_FEWSHOTS = [
    {"role": "user",      "content": "웹 서비스, 월 100만 MAU..."},
    {"role": "assistant", "content": "요구사항을 분석하고 핵심 쟁점을 정리하겠습니다..."},
]
COT_HINT = "Let's reason step-by-step and verify with relevant documents or tools before finalizing."
```

---

### ChromaDB + MiniLM

**역할 분리**

```
텍스트(문자열)
      ↓
  MiniLM          ← 텍스트를 숫자 벡터로 변환 (임베딩 모델)
      ↓
[0.23, -0.81, 0.45, ...]   ← 384차원 숫자 배열
      ↓
  ChromaDB        ← 벡터를 저장하고 유사한 것을 찾아줌 (벡터 데이터베이스)
```

- **MiniLM** = 번역기 (텍스트 → 벡터)
- **ChromaDB** = 도서관 (벡터 저장 + 유사 벡터 검색)

**MiniLM (all-MiniLM-L6-v2)**

| 항목 | 내용 |
|---|---|
| 제작 | Microsoft Research |
| 모델 크기 | 약 80MB |
| 출력 | 384차원 벡터 |
| 실행 방식 | ONNX 형식으로 로컬 CPU에서 실행 |
| 비용 | 무료 (API 호출 없음) |

**ChromaDB 스토리지 구조**

```
.chroma/                        ← 로컬 파일 시스템에 저장
├── chroma.sqlite3              ← 메타데이터 (원본 텍스트, 출처)
└── {UUID}/
    ├── data_level0.bin         ← 실제 벡터값 바이너리
    ├── header.bin              ← HNSW 인덱스 헤더
    ├── length.bin              ← 벡터 개수 정보
    └── link_lists.bin          ← HNSW 그래프 연결 정보
```

> `.chroma/` 폴더는 `python rag/ingest.py` 최초 실행 시 자동 생성됩니다.

**사용 위치**: `rag/ingest.py` (저장), `rag/retriever.py` (검색)

```python
# 저장 (ingest.py)
client = chromadb.PersistentClient(path=DB_DIR)
coll = client.get_or_create_collection("cloud_advisor")
coll.add(ids=[...], documents=["AWS 기본...", "Azure 기본..."], metadatas=[...])
# MiniLM이 텍스트 → 벡터 자동 변환 후 저장

# 검색 (retriever.py)
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
ef = DefaultEmbeddingFunction()   # MiniLM
coll = client.get_or_create_collection("cloud_advisor", embedding_function=ef)
res = coll.query(query_texts=["한국/일본 트래픽 CDN"], n_results=4)
# 1. 쿼리를 MiniLM으로 벡터화 → 2. 저장된 벡터와 거리 계산 → 3. 유사한 문서 4개 반환
```

---

### FastAPI + uvicorn

**역할 구분**

```
인터넷/브라우저
      ↓  HTTP 요청
  uvicorn           ← 웹 서버 (포트 열고 요청 수신)
      ↓  Python 함수 호출
  FastAPI           ← 프레임워크 (라우팅·검증·응답)
      ↓
  def advise(...)   ← 비즈니스 로직 (LangGraph 실행)
```

- **uvicorn**: 포트를 열고 HTTP 요청을 받는 ASGI 웹 서버
- **FastAPI**: 받은 요청을 검증·라우팅해서 Python 함수로 연결하는 프레임워크

**FastAPI 핵심 기능**

| 기능 | 설명 |
|---|---|
| 엔드포인트 | `@app.post("/advise")`로 URL과 함수 연결 |
| 자동 검증 | Pydantic 모델 선언만으로 요청 Body 자동 검증 |
| 자동 문서 | `/docs`에서 Swagger UI 자동 생성 |
| 타입 힌트 | 검증 + 문서화 + 자동완성 동시 제공 |

**사용 위치**: `api/main.py`

```python
app = FastAPI(title="AI Cloud Advisor API")
graph = build_graph()   # 서버 시작 시 한 번만 빌드, 이후 재사용

@app.post("/advise")
def advise(req: AdviseRequest) -> Dict[str, Any]:
    state = {
        "user_input": req.prompt,
        "monthly_users": req.monthly_users,
        ...
    }
    result = graph.invoke(state)    # LangGraph 4단계 실행
    return {"result": result.get("final")}
```

**uvicorn 실행 옵션**

```bash
uvicorn api.main:app --port 8000 --reload
#                               ↑
#                    코드 변경 감지 시 자동 재시작 (개발 전용)
```

---

### Streamlit

**Python 코드만으로 웹 앱을 만드는 프레임워크**입니다. HTML/CSS/JS 없이 AI 결과를 즉시 웹으로 시각화할 수 있어 AI 프로토타이핑에 최적화돼 있습니다.

**동작 원리**

```
streamlit run app_streamlit.py
      ↓
브라우저 접속 → 스크립트를 위→아래 순서로 실행해 화면 렌더링
      ↓
사용자가 값 변경 / 버튼 클릭
      ↓
스크립트 전체를 처음부터 다시 실행 (재렌더링)
```

**주요 컴포넌트**

| 컴포넌트 | 용도 |
|---|---|
| `st.sidebar` | 왼쪽 사이드바 영역 |
| `st.selectbox()` | 드롭다운 선택 |
| `st.number_input()` | 숫자 입력 |
| `st.text_area()` | 여러 줄 텍스트 입력 |
| `st.button()` | 버튼 (클릭 시 True 반환) |
| `st.spinner()` | 로딩 인디케이터 |
| `st.success/error/warning()` | 상태 메시지 |

**사용 위치**: `app_streamlit.py`

```python
with st.sidebar:
    region   = st.selectbox("Primary Region", ["apac", "us", "eu"])
    industry = st.selectbox("업종", ["general", "fintech", "ecommerce", ...])

user_prompt = st.text_area("Prompt", height=160)

if st.button("Get Recommendation", type="primary"):
    with st.spinner("Thinking..."):
        r = requests.post(api_url, json=payload, timeout=120)
        data = json.loads(r.json().get("result", "{}"))
        st.success(f"Recommended: **{data.get('provider')}**")
        st.subheader("Rationale"); st.write(data.get("rationale"))
```

---

### Pydantic

**Python 타입 힌트 기반의 데이터 검증 라이브러리**입니다. 외부에서 들어오는 데이터가 올바른 타입과 형식인지 자동으로 검증하고, 잘못됐을 때 명확한 에러를 반환합니다.

**검증 동작 방식**

```
요청 Body                    Pydantic 처리                결과
────────────                 ────────────                 ────
"monthly_users": "500"   →  str → int 자동 변환      →  500
"monthly_users": "abc"   →  변환 불가                →  422 에러 자동 반환
monthly_users 없음        →  기본값 100000 적용       →  100000
prompt 없음               →  필수값 누락              →  422 에러 자동 반환
```

**사용 위치**: `api/main.py`

```python
from pydantic import BaseModel, Field
from typing import Optional

class AdviseRequest(BaseModel):
    prompt:         str            = Field(..., description="User requirements")
    #                                      ↑ 필수값
    monthly_users:  Optional[int]  = 100000
    region:         Optional[str]  = "apac"
    storage_gb:     Optional[int]  = 200
    compute_tier:   Optional[str]  = "medium"
    industry:       Optional[str]  = "general"
    compliance:     Optional[str]  = None
    existing_cloud: Optional[str]  = None

# 엔드포인트에서 자동 검증
@app.post("/advise")
def advise(req: AdviseRequest):
    # req에 접근하는 시점엔 이미 검증 완료
    # 별도의 if/try 방어 코드 불필요
```

**Pydantic 없이 직접 구현했다면**

```python
# Pydantic 없는 경우 — 모든 파라미터를 수동 검증해야 함
if "prompt" not in body:
    raise HTTPException(400, "prompt is required")
try:
    monthly_users = int(body.get("monthly_users", 100000))
except (ValueError, TypeError):
    raise HTTPException(400, "monthly_users must be integer")
# ... 8개 파라미터 전부 반복
```

---

### python-dotenv

**`.env` 파일을 읽어 환경변수로 등록하는 라이브러리**입니다. API 키·비밀번호 같은 민감한 정보를 소스코드에 직접 쓰지 않고 별도 파일로 분리해 관리합니다.

**핵심 원칙**

```
소스코드에 직접 (위험)          python-dotenv 사용 (안전)
──────────────────────          ──────────────────────────
api_key = "sk-abc123..."        # .env 파일 (gitignore 제외)
↑ GitHub에 올라가면 키 노출      OPENAI_API_KEY=sk-abc123...

                                # 소스코드
                                load_dotenv()
                                api_key = os.getenv("OPENAI_API_KEY")
                                ↑ 코드엔 키가 없음
```

**우선순위**

```
시스템 환경변수  >  .env 파일  >  코드 기본값
    (높음)                           (낮음)
```

**.env 파일 구조**

```env
OPENAI_API_KEY=                      # 일반 OpenAI 키
AZURE_OPENAI_ENDPOINT=https://...    # Azure 엔드포인트 (있으면 Azure 우선 사용)
AZURE_OPENAI_API_KEY=...             # Azure API 키
MODEL_NAME=gpt-4o-mini               # 사용할 모델명
VECTOR_DB_DIR=.chroma                # ChromaDB 저장 경로
```

**사용 위치**: `utils/model.py`

```python
from dotenv import load_dotenv

load_dotenv()   # .env 파일 → 환경변수 등록 (파일 없어도 에러 없음)

def get_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(...)   # Azure 우선
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_model_name():
    return os.getenv("MODEL_NAME", "gpt-4o-mini")
    #                               ↑ 환경변수 없을 때 기본값
```

**환경별 운영 방식**

| 환경 | 방법 |
|---|---|
| 로컬 개발 | `.env` 파일 직접 작성 |
| Docker | `docker run --env-file .env` |
| 클라우드 (Azure/AWS) | 콘솔에서 환경변수 직접 등록 |
| GitHub Actions | Repository Secrets 설정 |

> `.env`는 `.gitignore`에 등록해 GitHub에 올라가지 않도록 반드시 제외합니다.
> `.env.example`은 팀 공유용 템플릿으로 GitHub에 포함합니다.

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
