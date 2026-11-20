"""검색: BM25(섹션 단위), 임베딩+FAISS(청크 단위→섹션), 하이브리드(RRF).

세 방식을 따로 평가할 수 있게 공개 메서드를 나눈다(scripts/eval_retrieval.py).
질의 임베딩은 저장·재사용하지 않는다(채점 요청 수집 금지, SPEC §8).
"""
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from .kb import Section
from .textnorm import tokenize

log = logging.getLogger("halt")
Ranked = List[Tuple[str, float]]  # (section_id, score), 점수 내림차순


class BM25:
    def __init__(self, docs: Sequence[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if self.n else 1.0
        self.post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.post[t].append((i, f))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for t, qf in Counter(query_tokens).items():
            p = self.post.get(t)
            if not p:
                continue
            idx = np.fromiter((i for i, _ in p), dtype=np.int64, count=len(p))
            tf = np.fromiter((f for _, f in p), dtype=np.float32, count=len(p))
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s


def rrf(rankings: Sequence[Ranked], k: int = 60) -> Ranked:
    acc: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, (sid, _) in enumerate(ranking):
            acc[sid] += 1.0 / (k + r + 1)
    return sorted(acc.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, sections: Dict[str, Section], kb_dir=None):
        self.sections = sections
        self.section_ids = list(sections)
        self.chunks = [c for s in self.section_ids for c in sections[s].chunks]
        self.index: Optional[faiss.Index] = None
        self.prep = None
        from .kbprep import KBPREP, KBPrep
        if KBPREP:
            # KB 전처리(v23~): 섹션·청크 BM25를 전처리 색인 글로 만든다. 기본값(v26)은 A-3·B-6·C-7이 꺼져 있어 v22 색인과 같고,
            # 켜면 표 문장·상투 문구 빼기·숫자 정규화 토큰이 들어간다.
            self.prep = KBPrep(sections, kb_dir)
            self.bm25 = BM25([self.prep._toks(self.prep.section_index_text[s]) for s in self.section_ids])
            self.bm25_chunks = BM25([self.prep._toks(t) for t in self.prep.chunk_index_text])
            self.prep.section_bm25, self.prep.chunk_bm25 = self.bm25, self.bm25_chunks
            return
        self.bm25 = BM25([tokenize(sections[s].text) for s in self.section_ids])
        # 청크 BM25(소제목 + 본문): 섹션 점수 = 그 섹션 청크 중 최고점. 튜터 인용 우선 섹션 고르기(TUTOR_CITE_CHUNK)에 쓴다.
        self.bm25_chunks = BM25([tokenize((c.heading or "") + "\n" + c.text) for c in self.chunks])

    def _qtoks(self, query: str) -> List[str]:
        return self.prep._toks(self.prep.query_text(query)) if self.prep else tokenize(query)

    # ---- BM25 ----
    def search_bm25(self, query: str, k: int = 10) -> Ranked:
        sc = self.bm25.scores(self._qtoks(query))
        order = np.argsort(-sc)[:k]
        return [(self.section_ids[i], float(sc[i])) for i in order if sc[i] > 0]

    def search_bm25_chunks(self, query: str, k: int = 10) -> Ranked:
        sc = self.bm25_chunks.scores(self._qtoks(query))
        best: Dict[str, float] = {}
        for i in np.argsort(-sc):
            if sc[i] <= 0 or len(best) >= k:
                break
            best.setdefault(self.chunks[i].section_id, float(sc[i]))
        return sorted(best.items(), key=lambda x: -x[1])

    # ---- 임베딩 ----
    @property
    def dense_ready(self) -> bool:
        return self.index is not None

    def set_embeddings(self, emb: np.ndarray) -> None:
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self.index = index

    def search_dense(self, qvec: np.ndarray, k: int = 10) -> Ranked:
        if self.index is None or qvec is None:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, min(len(self.chunks), k * 4))
        best: Dict[str, float] = {}
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            sid = self.chunks[i].section_id
            best[sid] = max(best.get(sid, -1.0), float(s))
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    # ---- 하이브리드 ----
    def search(self, query: str, qvec: Optional[np.ndarray], k: int = 5, mode: str = "hybrid") -> Ranked:
        if mode == "bm25" or qvec is None or not self.dense_ready:
            return self.search_bm25(query, k)
        if mode == "dense":
            return self.search_dense(qvec, k)
        return rrf([self.search_bm25(query, 20), self.search_dense(qvec, 20)])[:k]

    # ---- KB 임베딩 캐시 ----
    def chunk_digest(self, model: str) -> str:
        h = hashlib.sha256(model.encode())
        for c in self.chunks:
            h.update(c.chunk_id.encode() + b"\0" + c.text.encode() + b"\0")
        return h.hexdigest()


def load_cached_embeddings(cache_dir: Path, digest: str) -> Optional[np.ndarray]:
    meta, data = cache_dir / "kb_embeddings.json", cache_dir / "kb_embeddings.npy"
    try:
        if json.loads(meta.read_text()).get("digest") == digest:
            return np.load(data)
    except Exception:
        pass
    return None


def save_cached_embeddings(cache_dir: Path, digest: str, emb: np.ndarray, model: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "kb_embeddings.npy", emb)
    (cache_dir / "kb_embeddings.json").write_text(json.dumps(
        {"digest": digest, "model": model, "count": int(emb.shape[0]), "dim": int(emb.shape[1])}))
