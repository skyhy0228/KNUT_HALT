"""부하 테스트: 실제 채점 조건(튜터·가드 각 약 300건, 최대 64개 동시)을 공개 세트로 재현한다(개발용).

- 공개 세트 문항을 섞어 튜터 N건·가드 N건을 만들고(부족하면 반복), 동시 W개로 보낸다.
- 리다이렉트를 따라가지 않는다(채점기와 같음). 3xx·4xx·5xx·시간 초과·형식 오류를 따로 센다.
- 형식 검사는 배포 채점기(grader/scoring.py)의 parse_tutor/parse_guard를 그대로 쓴다.
- 문항·답변 내용은 출력하지 않는다. 시간·상태 코드·개수만 출력한다.

사용: python3 scripts/loadtest.py --url http://127.0.0.1:8000 [--n 300] [--workers 64] [--timeout 30]
토큰은 .env의 TEAM_TOKEN을 읽는다(출력하지 않음).
"""
import argparse
import json
import random
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402

FALLBACK_ANSWER = "지금은 답변을 만들기 어려워요. 궁금한 내용을 조금 더 구체적으로 다시 질문해 줄래요?"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(xs, q):
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=300, help="엔드포인트별 요청 수")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from dotenv import dotenv_values
    token = "".join(ch for ch in (dotenv_values(ROOT / ".env").get("TEAM_TOKEN") or "")
                    if ch.isprintable() and not ch.isspace())
    rng = random.Random(a.seed)
    jobs = []
    for ep in ("tutor", "guard"):
        items = read_jsonl(ROOT / f"public/{ep}.jsonl")
        pool = []
        while len(pool) < a.n:
            batch = items[:]
            rng.shuffle(batch)
            pool.extend(batch)
        jobs.extend((ep, it) for it in pool[:a.n])
    rng.shuffle(jobs)

    opener = urllib.request.build_opener(NoRedirect)
    lock = threading.Lock()
    inflight = [0, 0]  # 현재, 최대

    def one(job):
        ep, it = job
        payload = {"spec_version": "1.0", "session_id": "s_" + secrets.token_hex(8), "messages": it["messages"]}
        has_cand = "candidate_answer" in it
        if has_cand:
            payload["candidate_answer"] = it["candidate_answer"]
        req = urllib.request.Request(f"{a.url}/{ep}", data=json.dumps(payload, ensure_ascii=False).encode(),
                                     method="POST", headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {token}"})
        with lock:
            inflight[0] += 1
            inflight[1] = max(inflight[1], inflight[0])
        t0 = time.perf_counter()
        status, kind, fallback = None, "ok", False
        try:
            with opener.open(req, timeout=a.timeout) as r:
                status = r.status
                body = r.read(256 * 1024 + 1)
            if len(body) > 256 * 1024:
                kind = "too_large"
            else:
                obj = json.loads(body)
                parsed = scoring.parse_tutor(obj) if ep == "tutor" else scoring.parse_guard(obj, has_cand)
                if parsed is None:
                    kind = "invalid"
                elif ep == "tutor" and parsed["answer"] == FALLBACK_ANSWER:
                    fallback = True
        except urllib.error.HTTPError as e:
            status, kind = e.code, f"http_{e.code}"
        except (TimeoutError, OSError) as e:
            kind = "timeout" if "timed out" in str(e).lower() or isinstance(e, TimeoutError) else "conn_error"
        except json.JSONDecodeError:
            kind = "not_json"
        dt = time.perf_counter() - t0
        with lock:
            inflight[0] -= 1
        return ep, status, kind, dt, fallback

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    wall = time.perf_counter() - t0

    print(f"요청 {len(res)}건 (엔드포인트별 {a.n}), 동시 {a.workers} (실측 최대 동시 {inflight[1]}), 전체 {wall:.1f}초")
    for ep in ("tutor", "guard"):
        r = [x for x in res if x[0] == ep]
        lat = sorted(x[3] for x in r)
        kinds = Counter(x[2] for x in r)
        print(f"/{ep}: p50={pct(lat, .5):.2f}s p90={pct(lat, .9):.2f}s p95={pct(lat, .95):.2f}s "
              f"p99={pct(lat, .99):.2f}s max={lat[-1]:.2f}s  >25s={sum(t > 25 for t in lat)}  "
              f"상태={dict(Counter(x[1] for x in r))}  결과={dict(kinds)}"
              + (f"  폴백 답변={sum(x[4] for x in r)}" if ep == "tutor" else ""))
    bad = sum(x[2] != "ok" for x in res)
    print(f"오류 합계 {bad}건 (시간 초과·형식 오류·HTTP 오류 포함)")


if __name__ == "__main__":
    main()
