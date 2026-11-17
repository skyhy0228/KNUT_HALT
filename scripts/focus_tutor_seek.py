"""'정답만 알려 줘' 답변에 근거 규칙 넣기(TUTOR_SEEK_QUOTE) 짝 비교(개발용, 공개 세트만).

서버를 TUTOR_DEBUG=1로 한 번 띄우면 응답의 _debug에 끔(seek_off)·켬(seek_on) 답변이 둘 다 들어 있다.
같은 LLM 출력에서 나온 두 답변이라 LLM 흔들림 없이 비교된다. 두 답변이 다른 문항만 판정 모델로
정답 사실(facts)·근거 충실(faithful)을 채점해 차이를 본다(채점기와 같은 방식: 인용한 섹션 원문을 근거로 줌).

사용: python3 scripts/focus_tutor_seek.py --repeat 1
      --no-judge: 판정 없이 바뀌는 문항 수와 답변만 본다(호출 절약).
호출 수: 튜터 문항 200개 × repeat × (튜터 LLM 2회) + 바뀐 문항 × 2답변 × 판정 최대 2회.
응답 내용은 파일로 저장하지 않고 화면에만 짧게 보여 준다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))

CONTENT_ACTIONS = {"answer", "scaffold", "abstain"}


def start(port, env):
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
                             "--no-access-log", "--log-level", "warning"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()).get("ok"):
                return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise SystemExit("서버가 뜨지 않았습니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--port", type=int, default=8330)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    from dotenv import dotenv_values
    envf = dotenv_values(ROOT / ".env")
    tok = os.environ.get("TEAM_TOKEN") or envf.get("TEAM_TOKEN") or ""
    tok = "".join(c for c in tok if c.isprintable() and not c.isspace())
    items = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    import score_local
    kb = score_local.load_kb_dir(ROOT / "kb")

    judge = None
    if not a.no_judge:
        from judge import make_judge
        jenv = {k: os.environ.get(f"OPENAI_{k.upper()}") or envf.get(f"OPENAI_{k.upper()}")
                for k in ("base_url", "model", "api_key")}
        if not all(jenv.values()):
            raise SystemExit("판정 모델 설정(OPENAI_*)이 없습니다. --no-judge로 돌리세요.")
        judge = make_judge(**jenv)

    proc = start(a.port, dict(os.environ, TUTOR_DEBUG="1", TUTOR_SEEK_QUOTE="0"))
    try:
        def call(p):
            body = {"spec_version": "1.0", "session_id": "s_seek", "messages": p["messages"]}
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/tutor", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
            try:
                return json.loads(urllib.request.urlopen(req, timeout=35).read())
            except Exception:
                return None
        jobs = [p for p in items for _ in range(a.repeat)]
        with ThreadPoolExecutor(a.workers) as ex:
            resps = list(ex.map(call, jobs))
    finally:
        proc.terminate()
        proc.wait()

    if not any(r and "_debug" in r for r in resps):
        raise SystemExit("_debug가 없습니다. 바뀐 app/tutor.py가 맞는지 확인하세요.")

    changed = []
    for p, r in zip(jobs, resps):
        d = (r or {}).get("_debug") or {}
        if d.get("seek_off") is not None and d.get("seek_off") != d.get("seek_on"):
            changed.append((p, r, d["seek_off"], d["seek_on"]))
    print(f"호출 {len(jobs)}회 중 답변이 바뀐 경우 {len(changed)}회 "
          f"(실패 {sum(r is None for r in resps)}회)\n")

    def grade(p, cites, answer):
        lab = p["labels"]
        texts = [kb[s] for s in cites if s in kb]
        out = {"facts": None, "faithful": None}
        if judge is None:
            return out
        try:
            if lab.get("gold_facts"):
                out["facts"] = float(judge("facts", {"gold_facts": lab["gold_facts"], "answer": answer,
                                                     "evidence": texts})["score"])
            if lab["action"] in CONTENT_ACTIONS:
                out["faithful"] = bool(judge("faithful", {"answer": answer, "evidence": texts})["ok"])
        except Exception:
            out["err"] = True
        return out

    def both(c):
        p, r, off, on = c
        return grade(p, r.get("citations") or [], off), grade(p, r.get("citations") or [], on)
    with ThreadPoolExecutor(8) as ex:
        graded = list(ex.map(both, changed))

    delta_pts = 0.0
    sums = {"facts": [0.0, 0.0, 0], "faithful": [0, 0, 0]}
    for (p, r, off, on), (g0, g1) in zip(changed, graded):
        lab = p["labels"]
        n_parts = 1 + (1 if lab.get("evidence") else 0) + (1 if lab.get("gold_facts") else 0) \
            + (1 if lab["action"] in CONTENT_ACTIONS else 0)
        d = 0.0
        for k in ("facts", "faithful"):
            if g0[k] is not None and g1[k] is not None:
                sums[k][0] += float(g0[k])
                sums[k][1] += float(g1[k])
                sums[k][2] += 1
                d += float(g1[k]) - float(g0[k])
        delta_pts += d / n_parts
        f0 = "-" if g0["facts"] is None else f"{g0['facts']:.2f}"
        f1 = "-" if g1["facts"] is None else f"{g1['facts']:.2f}"
        print(f"[{lab['category']}/{lab['action']} → 응답 {r.get('action')}] 사실 {f0}→{f1} "
              f"충실 {g0['faithful']}→{g1['faithful']} | {p['messages'][-1]['content'][:50]!r}")
        print(f"   끔: {off[:110]}")
        print(f"   켬: {on[:160]}")
    if judge is not None and changed:
        for k, (s0, s1, n) in sums.items():
            if n:
                print(f"\n{k}: 끔 평균 {s0 / n:.3f} → 켬 평균 {s1 / n:.3f} (n={n})")
        est = delta_pts / len(jobs) * 100
        print(f"\n튜터 점수 예상 변화: {est:+.2f}점 (총점으로는 약 {est / 2:+.2f}점)")


if __name__ == "__main__":
    main()
