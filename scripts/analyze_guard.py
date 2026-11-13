"""가드 항목별 분석(개발용, 공개 세트만): crossval.py --dump 결과(pred.jsonl)를 배포 채점기로 다시 채점해
block F1·범주 정확도·grounded·구간 F1·safe를 실행별로, 여러 실행이면 평균과 표준편차로 보여 준다.

사용: python3 scripts/analyze_guard.py <run1>/pred.jsonl [<run2>/pred.jsonl ...] [--label 이름]
문항·답변 내용은 출력하지 않는다.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grader"))
import score_local  # noqa: E402
import scoring  # noqa: E402


def key(msgs, cand):
    return json.dumps(msgs, ensure_ascii=False, sort_keys=True) + str(cand)


def metrics(pred_path, public):
    g = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["endpoint"] != "guard":
            continue
        p = public[key(d["messages"], d.get("candidate_answer"))]
        item, v = score_local.as_scoring_input(p)
        g.append(scoring.score_guard(item, v, d["resp"]))
    tp = sum(r["gold_block"] and r["input_correct"] for r in g)
    fn = sum(r["gold_block"] and not r["input_correct"] for r in g)
    fp = sum(not r["gold_block"] and not r["input_correct"] for r in g)
    cs = [r for r in g if r["category_scored"]]
    o = [r for r in g if r["has_output"]]
    return {
        "n": len(g), "guard": scoring.guard_score(g), "block_f1": 2 * tp / (2 * tp + fp + fn),
        "missed": fn, "false_block": fp, "category": sum(r["category_correct"] for r in cs) / len(cs),
        "grounded": sum(r["output_correct"] for r in o) / len(o), "span_f1": sum(r["span_f1"] for r in o) / len(o),
        "safe": sum(r["safe_correct"] for r in o) / len(o),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    public = {key(p["messages"], p.get("candidate_answer")): p for p in score_local._read_jsonl(ROOT / "public/guard.jsonl")}
    rows = [metrics(p, public) for p in a.preds]
    cols = ["guard", "block_f1", "missed", "false_block", "category", "grounded", "span_f1", "safe"]
    for p, r in zip(a.preds, rows):
        print(f"{a.label} {Path(p).parent.name}: " + "  ".join(f"{c}={r[c]:.3f}" if isinstance(r[c], float) else f"{c}={r[c]}" for c in cols))
    if len(rows) > 1:
        print(f"{a.label} 평균±표준편차: " + "  ".join(
            f"{c}={statistics.mean(r[c] for r in rows):.3f}±{statistics.stdev(r[c] for r in rows):.3f}" for c in cols))


if __name__ == "__main__":
    main()
