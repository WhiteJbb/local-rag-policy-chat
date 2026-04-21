"""
main.py
RAG 파이프라인 실행 진입점.

사용법:
  # 1) 최초 인덱싱
  python main.py --index --docs ./docs

  # 2) 질문 (인덱싱 이후)
  python main.py --query "연차 3일 신청 가능한가요?"

  # 3) 인터랙티브 모드
  python main.py --interactive
"""

import argparse

from document_loader import load_directory, chunk_adaptive
from vector_store import index_documents, load_vector_store, get_retriever
from rag_chain import build_rag_chain, ask


def run_index(docs_dir: str, recreate: bool = False):
    """문서를 로드 → 청킹 → Qdrant 인덱싱"""
    print(f"\n{'='*60}")
    print(f"[STEP 1] 문서 로드: {docs_dir}")
    raw_docs = load_directory(docs_dir)

    print(f"\n[STEP 2] 청킹")
    chunks = chunk_adaptive(raw_docs)

    print(f"\n[STEP 3] Qdrant 인덱싱 (recreate={recreate})")
    index_documents(chunks, recreate=recreate)
    print("\n인덱싱 완료!")


def run_query(question: str):
    """저장된 벡터 스토어에서 검색 후 답변 생성"""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, top_k=6)
    chain = build_rag_chain(retriever)
    ask(chain, question)


def run_interactive():
    """대화형 모드: 종료 시 'exit' 입력"""
    print("\n[인터랙티브 RAG 모드] 종료: 'exit' 입력")
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, top_k=6)
    chain = build_rag_chain(retriever)

    while True:
        q = input("\n질문 > ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        ask(chain, q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="사내 규정 RAG 파이프라인")
    parser.add_argument("--index", action="store_true", help="문서 인덱싱 실행")
    parser.add_argument("--docs", type=str, default="./docs", help="문서 디렉터리 경로")
    parser.add_argument("--recreate", action="store_true", help="컬렉션 재생성 (문서 교체 시)")
    parser.add_argument("--query", type=str, help="단일 질문 실행")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    args = parser.parse_args()

    if args.index:
        run_index(args.docs, recreate=args.recreate)
    elif args.query:
        run_query(args.query)
    elif args.interactive:
        run_interactive()
    else:
        parser.print_help()
