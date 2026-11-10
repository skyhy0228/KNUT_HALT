"""튜터 오답 분석(개발용, 공개 세트만): crossval.py --dump 결과를 항목별로 다시 채점한다.

- pred.jsonl(공개 세트 문항에 대한 내 서버 응답)을 공개 세트 라벨과 맞춰
  배포 채점기(grader/scoring.py)의 score_tutor로 action·인용·정답 사실·근거 충실·응함 여부를 나눠 본다.
- 결과는 --out(JSONL)에 쓰고, 화면에는 집계만 출력한다.

사용: python3 scripts/analyze_tutor.py --pred <tmp>/pred.jsonl --out <tmp>/tutor_breakdown.jsonl [--judge]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grader"))
import score_local  # noqa: E402
import scoring  # noqa: E402
from judge import make_judge  # noqa: E402


def key(msgs):
    return json.dumps(msgs, ensure_ascii=False, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    from dotenv import dotenv_values
    env_file = dotenv_values(ROOT / ".env")
    judge = None
    if a.judge:
        jenv = {k: os.environ.get(f"OPENAI_{k.upper()}") or env_file.get(f"OPENAI_{k.upper()}")
                for k in ("base_url", "model", "api_key")}
        judge = make_judge(**jenv)

    public = {key(p["messages"]): p for p in score_local._read_jsonl(ROOT / "public/tutor.jsonl")}
    preds = [json.loads(l) for l in Path(a.pred).read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = [d for d in preds if d["endpoint"] == "tutor"]
    kb = score_local.load_kb_dir(ROOT / "kb")

    def one(d):
        p = public[key(d["messages"])]
        item, v = score_local.as_scoring_input(p)
        r = scoring.score_tutor(item, v, d["resp"], judge or score_local._no_judge, kb)
        lab = p["labels"]
        return {"id": p["id"], "category": lab["category"], "gold_action": lab["action"],
                "pred_action": (d["resp"] or {}).get("action"), "evidence": lab.get("evidence") or [],
                "cited": (d["resp"] or {}).get("citations"), "points": round(scoring.tutor_points(r), 3), **r}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, preds))
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    n = len(rows)
    ok = [r for r in rows if not r["judge_error"]]
    print(f"튜터 {n}건, 판정 오류 {n - len(ok)}건, 평균 {100 * sum(r['points'] for r in ok) / max(1, len(ok)):.2f}")
    print(f"action 정확도 {sum(r['action_correct'] for r in rows)}/{n}")
    cit = [r for r in rows if r["citation_p"] is not None]
    f1 = [2 * r["citation_p"] * r["citation_r"] / (r["citation_p"] + r["citation_r"])
          if r["citation_p"] + r["citation_r"] else 0 for r in cit]
    print(f"인용 F1 평균 {sum(f1) / max(1, len(f1)):.3f} (근거 있는 문항 {len(cit)}건)")
    fa = [r["facts"] for r in ok if r["facts"] is not None]
    if fa:
        print(f"정답 사실 평균 {sum(fa) / len(fa):.3f} ({len(fa)}건)")
    fu = [r["faithful"] for r in ok if r["faithful"] is not None]
    if fu:
        print(f"근거 충실 {sum(fu)}/{len(fu)}")
    co = [r for r in ok if r["complied"] is not None]
    if co:
        print(f"공격 문항 응함(0점) {sum(r['complied'] for r in co)}/{len(co)}")
    conf = Counter((r["gold_action"], r["pred_action"]) for r in rows if not r["action_correct"])
    print("action 오답(정답→예측):", dict(conf.most_common()))
    lost = defaultdict(float)
    for r in ok:
        lost[r["category"]] += 1 - r["points"]
    print("범주별 잃은 점수(문항 수 환산):", {k: round(v, 2) for k, v in sorted(lost.items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
