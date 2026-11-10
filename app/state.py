"""서버 시작 시 준비하는 공유 상태(KB, 색인, LLM 클라이언트).

색인·임베딩은 첫 요청 전에 모두 만든다(SPEC §3). KB 임베딩은 cache/에 저장해 재시작 때 재사용한다.
임베딩 API가 실패해도 서버는 BM25만으로 뜬다.
"""
import logging
import time
from dataclasses import dataclass

from .config import Settings
from .examples import ExamplePool
from .kb import load_kb
from .llm import LLM
from .retrieval import Retriever, load_cached_embeddings, save_cached_embeddings

log = logging.getLogger("halt")


@dataclass
class AppState:
    settings: Settings
    retriever: Retriever
    llm: LLM
    examples: ExamplePool


async def build_state(settings: Settings) -> AppState:
    t0 = time.perf_counter()
    sections = load_kb(settings.kb_dir)
    retriever = Retriever(sections)
    llm = LLM(settings)
    log.info("kb_loaded sections=%d chunks=%d", len(sections), len(retriever.chunks))

    if settings.use_dense:
        digest = retriever.chunk_digest(settings.embedding_model)
        emb = load_cached_embeddings(settings.cache_dir, digest)
        source = "cache"
        if emb is None or emb.shape[0] != len(retriever.chunks):
            source = "api"
            try:
                emb = await llm.embed_texts([c.text for c in retriever.chunks])
                save_cached_embeddings(settings.cache_dir, digest, emb, settings.embedding_model)
            except Exception as e:
                emb = None
                log.warning("kb_embedding_failed type=%s -> BM25 only", type(e).__name__)
        if emb is not None:
            retriever.set_embeddings(emb)
            log.info("kb_embeddings_ready source=%s shape=%s", source, tuple(emb.shape))

    examples = ExamplePool(settings.fewshot_files)
    log.info("startup_ready dense=%s elapsed_ms=%d", retriever.dense_ready, (time.perf_counter() - t0) * 1000)
    return AppState(settings, retriever, llm, examples)
