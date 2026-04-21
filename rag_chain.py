"""
rag_chain.py
LangChain LCEL 패턴으로 RAG 체인을 구성합니다.
Ollama gemma4-e2b 를 로컬 LLM으로 사용합니다.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama


# ─────────────────────────────────────────
# LLM 설정
# ─────────────────────────────────────────
OLLAMA_MODEL = "gemma4:e2b"   # ollama run gemma4:e2b 로 먼저 pull 필요
OLLAMA_BASE_URL = "http://localhost:11434"


def get_llm(model: str = OLLAMA_MODEL, temperature: float = 0.0) -> ChatOllama:
    """
    Ollama 로컬 LLM을 반환합니다.
    temperature=0 으로 고정해 규정 답변의 일관성을 높입니다.
    """
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ─────────────────────────────────────────
# 프롬프트 템플릿
# ─────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
당신은 사내 규정 전문 어시스턴트입니다.
아래 제공된 [관련 규정] 내용만을 근거로 질문에 답하세요.

규정에 명시되지 않은 내용은 절대 추측하거나 임의로 답하지 마세요.
근거가 없으면 "해당 내용은 제공된 규정에서 확인할 수 없습니다."라고 답하세요.

[관련 규정]
{context}

[질문]
{question}

[답변]
""")


# ─────────────────────────────────────────
# 컨텍스트 포맷터
# ─────────────────────────────────────────
def format_context(docs: List[Document]) -> str:
    """
    검색된 청크들을 프롬프트에 넣을 문자열로 변환합니다.
    출처(파일명, 페이지/슬라이드)를 함께 포함합니다.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = meta.get("source", "알 수 없음")
        loc = meta.get("page") or meta.get("slide") or "-"
        loc_label = "페이지" if meta.get("page") else "슬라이드"
        parts.append(
            f"[출처 {i}: {source} {loc_label} {loc}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────
# LCEL RAG 체인 빌드
# ─────────────────────────────────────────
NO_CONTEXT_ANSWER = "검색된 관련 규정이 없습니다. 질문을 구체적으로 바꾸거나 관련 문서가 인덱싱되어 있는지 확인해 주세요."


def build_rag_chain(retriever):
    """
    LCEL 파이프라인:
      질문 → retriever → (빈 결과면 즉시 반환) → LLM → 답변

    반환값: { "answer": str, "context": List[Document] }
    """
    llm = get_llm()

    setup = RunnableParallel(
        context=retriever | format_context,
        question=RunnablePassthrough(),
        docs=retriever,
    )

    def route(inputs: dict) -> dict:
        if not inputs["docs"]:
            return {"answer": NO_CONTEXT_ANSWER, "context": []}
        answer = (RAG_PROMPT | llm | StrOutputParser()).invoke({
            "context": inputs["context"],
            "question": inputs["question"],
        })
        return {"answer": answer, "context": inputs["docs"]}

    return setup | RunnableLambda(route)


# ─────────────────────────────────────────
# 편의 함수: 질문 한 번에 실행
# ─────────────────────────────────────────
def ask(chain, question: str) -> dict:
    """
    질문을 입력하면 답변과 출처 청크를 반환합니다.

    Returns:
        {
            "answer": "답변 텍스트",
            "context": [Document, ...]   # 근거 청크 리스트
        }
    """
    result = chain.invoke(question)

    W = 64
    print(f"\n┌{'─'*W}┐")
    print(f"│ Q  {question[:W-4]:<{W-4}} │")
    print(f"├{'─'*W}┤")
    for line in result["answer"].splitlines():
        while len(line) > W - 4:
            print(f"│    {line[:W-4]} │")
            line = line[W-4:]
        print(f"│    {line:<{W-4}} │")
    print(f"└{'─'*W}┘")

    ctx = result["context"]
    if not ctx:
        print("\n  (검색된 근거 없음)")
        return result

    print(f"\n  📎 근거 {len(ctx)}건")
    for i, doc in enumerate(ctx, 1):
        meta = doc.metadata
        src = meta.get("source", "?")
        if meta.get("section"):
            loc_str = f"# {meta['section']}"
        elif meta.get("page"):
            loc_str = f"p.{meta['page']}"
        elif meta.get("slide"):
            loc_str = f"slide {meta['slide']}"
        else:
            loc_str = ""

        header = f"  [{i}] {src}  {loc_str}"
        print(f"\n{header}")
        print(f"  {'─'*min(W, len(header)+4)}")
        preview = doc.page_content.strip().replace("\n", " ")
        print(f"  {preview[:200]}{'...' if len(preview) > 200 else ''}")

    print()
    return result
