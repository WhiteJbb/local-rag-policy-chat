"""
app.py
Chainlit 기반 RAG 채팅 UI.
실행: chainlit run app.py
"""

import chainlit as cl

from vector_store import load_vector_store, get_retriever
from rag_chain import build_rag_chain, NO_CONTEXT_ANSWER

# 앱 시작 시 한 번만 로드 — 세션마다 재로드 방지
_chain = None
_init_error = None

try:
    _chain = build_rag_chain(get_retriever(load_vector_store(), top_k=6))
except Exception as e:
    _init_error = str(e)


@cl.on_chat_start
async def on_chat_start():
    if _init_error:
        await cl.Message(
            content=f"❌ 로딩 실패: {_init_error}\n\n먼저 `python main.py --index --docs ./docs` 를 실행해주세요."
        ).send()
    else:
        await cl.Message(content="✅ 준비 완료! 사내 규정에 대해 질문하세요.").send()


@cl.on_message
async def on_message(message: cl.Message):
    if _chain is None:
        await cl.Message(content="초기화에 실패했습니다. 터미널 로그를 확인해주세요.").send()
        return

    question = message.content.strip()
    if not question:
        return

    thinking = cl.Message(content="")
    await thinking.send()

    result = await cl.make_async(_chain.invoke)(question)
    answer: str = result["answer"]
    docs: list = result["context"]

    await thinking.remove()

    elements = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src = meta.get("source", "?")

        if meta.get("section"):
            loc = f"#{meta['section']}"
        elif meta.get("page"):
            loc = f"p.{meta['page']}"
        elif meta.get("slide"):
            loc = f"slide {meta['slide']}"
        else:
            loc = ""

        label = f"[{i}] {src}  {loc}"
        elements.append(cl.Text(name=label, content=doc.page_content.strip(), display="side"))

    no_source = answer == NO_CONTEXT_ANSWER
    await cl.Message(
        content=answer,
        elements=elements if not no_source else [],
    ).send()
