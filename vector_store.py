"""
vector_store.py
Qdrant 로컬 인스턴스에 임베딩을 저장하고 유사도 검색을 수행합니다.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"   # Qdrant 로컬 주소
COLLECTION_NAME = "company_regulations"

# 한국어 경량 테스트용 모델 (~400MB, dim=768)
# 고품질이 필요하면 intfloat/multilingual-e5-large 로 교체 후 --recreate 재인덱싱
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # 고품질 한국어 (dim=1024)

EMBEDDING_DIM = {
    "jhgan/ko-sroberta-multitask": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
}


# ─────────────────────────────────────────
# 임베딩 모델 로드
# ─────────────────────────────────────────
def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """HuggingFace 임베딩 모델을 로컬에서 로드합니다."""
    print(f"임베딩 모델 로딩: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # GPU 없는 환경
        encode_kwargs={"normalize_embeddings": True},
    )


# ─────────────────────────────────────────
# Qdrant 컬렉션 초기화
# ─────────────────────────────────────────
def init_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
    recreate: bool = False,
) -> None:
    """
    Qdrant 컬렉션을 생성합니다.
    recreate=True 이면 기존 컬렉션을 삭제 후 재생성 (Replace 전략).
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name)
            print(f"기존 컬렉션 삭제: {collection_name}")
        else:
            print(f"컬렉션 이미 존재: {collection_name} (재사용)")
            return

    dim = EMBEDDING_DIM.get(model_name, 384)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"컬렉션 생성 완료: {collection_name} (dim={dim})")


# ─────────────────────────────────────────
# 문서 인덱싱
# ─────────────────────────────────────────
def index_documents(
    docs: List[Document],
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
    recreate: bool = False,
) -> QdrantVectorStore:
    """
    청크 문서 리스트를 임베딩하여 Qdrant에 저장합니다.

    Args:
        docs: 청킹된 Document 리스트
        collection_name: Qdrant 컬렉션 이름
        model_name: 임베딩 모델 이름
        recreate: True 면 기존 컬렉션 삭제 후 재인덱싱 (문서 교체 시 사용)
    """
    client = QdrantClient(url=QDRANT_URL)
    init_collection(client, collection_name, model_name, recreate)

    embeddings = get_embeddings(model_name)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # 배치로 나눠서 추가 (메모리 절약)
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"인덱싱 중... {min(i + batch_size, len(docs))}/{len(docs)}")

    print(f"\n인덱싱 완료: {len(docs)}개 청크 → Qdrant [{collection_name}]")
    return vector_store


# ─────────────────────────────────────────
# 벡터 스토어 로드 (인덱싱 이후 재사용)
# ─────────────────────────────────────────
def load_vector_store(
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> QdrantVectorStore:
    """이미 인덱싱된 컬렉션을 불러옵니다."""
    client = QdrantClient(url=QDRANT_URL)
    embeddings = get_embeddings(model_name)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ─────────────────────────────────────────
# 검색 (retriever 생성)
# ─────────────────────────────────────────
def get_retriever(
    vector_store: QdrantVectorStore,
    top_k: int = 6,
    score_threshold: Optional[float] = 0.4,
):
    """
    유사도 기반 retriever를 반환합니다.

    top_k: 반환할 청크 수 (사내 규정 복합 조건 → 5~8 권장)
    score_threshold: 유사도 최소 임계값 (낮은 점수 청크 필터링)
    """
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k,
            "score_threshold": score_threshold,
        },
    )
