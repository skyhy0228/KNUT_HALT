"""공개 튜터 세트로 BM25 / 임베딩 / 하이브리드 검색의 Recall@k를 따로 측정한다(개발용).

사용: python3 scripts/eval_retrieval.py [--no-dense]
공개 세트(배포 자료)만 사용한다. 질의 임베딩은 이 실행 중에만 쓰고 저장하지 않는다.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.state import build_state  # noqa: E402
from app.textnorm import build_dense_query, build_query  # noqa: E402

KS = (1, 3, 5, 10)


async def main(use_dense: bool):
    settings = load_settings()
    state = await build_state(settings)
    r = state.retriever
    items = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [it for it in items if it["labels"].get("evidence")]

    qvecs = [None] * len(items)
    if use_dense and r.dense_ready:
        emb = await state.llm.embed_texts([build_dense_query(it["messages"]) for it in items])
        qvecs = list(emb)

    modes = ["bm25"] + (["dense", "hybrid"] if use_dense and r.dense_ready else [])
    for mode in modes:
        hits = {k: 0 for k in KS}
        for it, qv in zip(items, qvecs):
            gold = set(it["labels"]["evidence"])
            ranked = [sid for sid, _ in r.search(build_query(it["messages"]), qv, k=max(KS), mode=mode)]
            for k in KS:
                hits[k] += bool(gold & set(ranked[:k]))
        print(f"{mode:7s} " + "  ".join(f"R@{k}={hits[k] / len(items):.3f}" for k in KS) + f"  (n={len(items)})")
    await state.llm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    asyncio.run(main(not ap.parse_args().no_dense))
