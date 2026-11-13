"""공개 세트 2-fold 교차 검증(개발용).

few-shot 예시 풀에 채점 문항 자신(과 같은 묶음)이 들어가면 로컬 점수가 부풀려진다.
그래서 공개 세트를 문항 ID 해시로 A/B로 나누고, A를 예시 풀로 쓴 서버로 B를 채점(그 반대도)한다.

사용: python3 scripts/crossval.py [--judge] [--workers 64] [--port 8100]
  --judge: .env의 OPENAI_* 로 판정 모델을 켠다(LLM 호출이 문항당 최대 3회 늘어남).
분할 파일은 --tmp 디렉터리(기본: 시스템 임시 폴더)에 쓰고, 문항·답변 내용은 출력하지 않는다.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grader"))


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def fold_of(item_id: str) -> str:
    return "A" if int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def group_folds(items, threshold=0.5):
    """마지막 학습자 발화의 문자 3-gram 유사도가 threshold 이상인 문항을 같은 그룹으로 묶고(합집합),
    그룹 단위로 A/B를 나눈다. 거의 같은 문항이 양쪽에 갈려 few-shot 예시로 새는 것을 막는다."""
    import re
    sys.path.insert(0, str(ROOT))
    from app.textnorm import normalize

    def grams(t):
        s = re.sub(r"\s+", "", normalize(t))
        return {s[i:i + 3] for i in range(len(s) - 2)}
    G = [grams(it["messages"][-1]["content"]) for it in items]
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if G[i] and G[j] and len(G[i] & G[j]) / len(G[i] | G[j]) >= threshold:
                parent[find(i)] = find(j)
    root_id = {}
    for i, it in enumerate(items):
        r = find(i)
        root_id[r] = min(root_id.get(r, it["id"]), it["id"])
    return {it["id"]: fold_of(root_id[find(i)]) for i, it in enumerate(items)}


def wait_health(port, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--dump", action="store_true",
                    help="공개 세트 문항별 응답을 --tmp/pred.jsonl에 저장(오류 분석용, 공개 세트만)")
    ap.add_argument("--endpoints", default="tutor,guard", help="채점할 엔드포인트(예: guard). 예시 풀은 항상 둘 다 쓴다")
    ap.add_argument("--group-split", action="store_true", help="유사 질문 그룹 단위로 A/B 분할")
    a = ap.parse_args()
    endpoints = [e.strip() for e in a.endpoints.split(",") if e.strip()]
    dumped = []

    from dotenv import dotenv_values
    import score_local
    from judge import make_judge

    env_file = dotenv_values(ROOT / ".env")
    token = "".join(ch for ch in (env_file.get("TEAM_TOKEN") or "") if ch.isprintable() and not ch.isspace())
    tmp = Path(a.tmp or tempfile.mkdtemp(prefix="halt_cv_"))
    folds = {"A": {}, "B": {}}
    all_items = {ep: read_jsonl(ROOT / f"public/{ep}.jsonl") for ep in ("tutor", "guard")}
    assign = (group_folds(all_items["tutor"] + all_items["guard"]) if a.group_split
              else {it["id"]: fold_of(it["id"]) for ep in all_items for it in all_items[ep]})
    for ep in ("tutor", "guard"):
        for f in "AB":
            folds[f][ep] = []
        for it in all_items[ep]:
            folds[assign[it["id"]]][ep].append(it)
    for f in "AB":
        d = tmp / f
        d.mkdir(parents=True, exist_ok=True)
        for ep in ("tutor", "guard"):
            (d / f"{ep}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in folds[f][ep]) + "\n",
                                           encoding="utf-8")

    judge = None
    if a.judge:
        jenv = {k: os.environ.get(f"OPENAI_{k.upper()}") or env_file.get(f"OPENAI_{k.upper()}")
                for k in ("base_url", "model", "api_key")}
        judge = make_judge(**jenv) if all(jenv.values()) else None
        if judge is None:
            print("판정 모델 설정이 없어 판정 없이 진행합니다.")

    kb = score_local.load_kb_dir(ROOT / "kb")
    results = []
    for pool, test in (("A", "B"), ("B", "A")):
        env = dict(os.environ, FEWSHOT_FILES=f"{tmp / pool / 'tutor.jsonl'},{tmp / pool / 'guard.jsonl'}")
        log_path = tmp / f"server_{test}.log"
        with open(log_path, "w") as lf:
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
                                     "--port", str(a.port), "--loop", "uvloop", "--no-access-log"],
                                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            if not wait_health(a.port):
                print("서버 시작 실패:", log_path)
                return
            url = f"http://127.0.0.1:{a.port}"
            opener = urllib.request.build_opener()

            def call(endpoint, payload):
                req = urllib.request.Request(f"{url}/{endpoint}", data=json.dumps(payload, ensure_ascii=False).encode(),
                                             method="POST", headers={"Content-Type": "application/json",
                                                                     "Authorization": f"Bearer {token}"})
                with opener.open(req, timeout=30) as r:
                    resp = json.loads(r.read(256 * 1024 + 1))
                if a.dump:
                    dumped.append({"endpoint": endpoint, "messages": payload["messages"],
                                   "candidate_answer": payload.get("candidate_answer"), "resp": resp})
                return resp

            public = [it for ep in endpoints for it in folds[test][ep]]
            t0 = time.time()
            out = score_local.score_local(public, call, judge, kb, workers=a.workers)
            dt = time.time() - t0
            results.append(out)
            print(f"[예시 풀 {pool} → 채점 {test}] n={len(public)} {dt:.1f}초  총점 {out['total']} "
                  f"(튜터 {out['tutor']}, 가드 {out['guard']}) 형식 오류 {out['invalid']} 판정 오류 {out['judge_errors']}")
            print("  범주별:", json.dumps(out["per_category"], ensure_ascii=False))
            import re as _re
            for ep in ("tutor", "guard"):
                ms = sorted(int(m.group(1)) for m in (_re.search(rf"path=/{ep} status=\d+ ms=(\d+)", l)
                                                       for l in log_path.read_text().splitlines()) if m)
                if ms:
                    print(f"  /{ep} 지연: p50={ms[len(ms) // 2]}ms p95={ms[int(len(ms) * .95)]}ms max={ms[-1]}ms")
            warns = [l.split("WARNING ", 1)[1].strip() for l in log_path.read_text().splitlines() if "WARNING" in l]
            if warns:
                from collections import Counter
                print("  서버 경고:", dict(Counter(w.split()[0] for w in warns)))
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    if a.dump:
        (tmp / "pred.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in dumped) + "\n",
                                        encoding="utf-8")
        print("문항별 응답 저장:", tmp / "pred.jsonl")
    if len(results) == 2:
        avg = {k: round(sum(r[k] for r in results) / 2, 2) for k in ("total", "tutor", "guard")}
        print(f"[평균] 총점 {avg['total']} (튜터 {avg['tutor']}, 가드 {avg['guard']})"
              + (" — 판정 모델 없음: 튜터는 action·인용만" if judge is None else ""))


if __name__ == "__main__":
    main()
