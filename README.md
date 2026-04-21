# 로컬 RAG 파이프라인 (Ollama + Qdrant + LangChain + Chainlit)

## 구성

| 역할 | 기술 |
|------|------|
| LLM | Ollama `gemma4:e2b` (로컬) |
| 임베딩 | `jhgan/ko-sroberta-multitask` (한국어, dim=768) |
| 벡터 DB | Qdrant (로컬 Docker) |
| 문서 형식 | PDF / PPTX / MD |
| 프레임워크 | LangChain LCEL |
| UI | Chainlit |

---

## 파일 구조

```
localRAG/
├── main.py              # CLI 진입점 (인덱싱 / 질문 / 인터랙티브)
├── app.py               # Chainlit 웹 UI
├── document_loader.py   # PDF / PPTX / MD 로드 + 청킹
├── vector_store.py      # Qdrant 인덱싱 / 검색
├── rag_chain.py         # LangChain LCEL 체인 + 프롬프트
├── clean_md.py          # MD 파일 전처리 스크립트
├── docker-compose.yml   # Qdrant 컨테이너 설정
├── requirements.txt
└── docs/                # 여기에 규정 문서 넣기 (PDF / PPTX / MD)
```

---

## 실행 순서

### 1. 사전 준비

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Qdrant 실행 (Docker)

```bash
docker compose up -d
```

### 3. Ollama 실행 (로컬 설치본 사용)

```bash
ollama serve   # 이미 실행 중이면 스킵
```

### 4. 문서 준비

`docs/` 폴더에 PDF, PPTX, MD 파일을 넣습니다.

MD 파일의 경우 PDF 추출본이 지저분하다면 전처리 스크립트를 먼저 실행하세요.

```bash
python clean_md.py "docs/파일명.md"
# → docs/파일명_clean.md 생성
```

### 5. 인덱싱

```bash
# 최초 인덱싱
python main.py --index --docs ./docs

# 문서 교체 / 임베딩 모델 변경 후 재인덱싱
python main.py --index --docs ./docs --recreate
```

### 6. 웹 UI 실행

```bash
chainlit run app.py
# → http://localhost:8000
```

### 7. CLI로 질문 (선택)

```bash
# 단일 질문
python main.py --query "연차는 며칠까지 신청할 수 있나요?"

# 대화형 모드
python main.py --interactive
```

---

## 주요 파라미터

| 파일 | 변수 | 설명 |
|------|------|------|
| `vector_store.py` | `EMBEDDING_MODEL` | 임베딩 모델 (기본: `jhgan/ko-sroberta-multitask`) |
| `vector_store.py` | `score_threshold` | 유사도 임계값 (기본 0.4, 검색이 안 되면 낮춤) |
| `vector_store.py` | `top_k` | 검색 청크 수 (기본 6) |
| `rag_chain.py` | `OLLAMA_MODEL` | Ollama 모델 변경 |
| `document_loader.py` | `MIN_CHUNK` | 청크 병합 기준 최소 크기 (기본 150자) |
| `document_loader.py` | `MAX_CHUNK` | 청크 분할 기준 최대 크기 (기본 800자) |
| `document_loader.py` | `OVERLAP` | 분할 시 오버랩 크기 (기본 150자) |

---

## 청킹 전략

`chunk_adaptive()` 방식을 사용합니다.

- **150자 미만** 청크 → 다음 청크와 병합 (목차·짧은 헤더 제거)
- **150 ~ 800자** 청크 → 그대로 유지
- **800자 초과** 청크 → `RecursiveCharacterTextSplitter`로 재분할

| | 개선 전 | 개선 후 |
|---|---|---|
| 청크 수 | 114개 | 78개 |
| 최솟값 | 20자 | 150자 |
| 최댓값 | 5,281자 | 797자 |
| 평균 | 280자 | 442자 |
| 쓸모없는 소형 청크 | 81개 | 0개 |

---

## 임베딩 모델 교체

`vector_store.py` 상단에서 변경:

```python
# 한국어 경량 (현재 기본값, ~400MB)
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# 고품질 다국어 (~1GB)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
```

> 모델 교체 후 반드시 `--recreate` 플래그로 재인덱싱 필요

---

## 동작 방식

```
docs/ (PDF / PPTX / MD)
    ↓ document_loader.py
  페이지 / 슬라이드 / 헤더 단위 로드
    ↓ chunk_adaptive()
  150~800자 균일 청크 (소형 병합 + 대형 분할)
    ↓ vector_store.py → Qdrant (dim=768)
  임베딩 저장
    ↓ (질문 시) score_threshold=0.4 필터 + top_k=6 검색
  관련 청크
    ↓ 빈 결과 → LLM 호출 없이 즉시 "검색 결과 없음" 반환
    ↓ 결과 있음 → rag_chain.py → Ollama LLM
  답변 + 출처 (Chainlit 사이드 패널)
```
