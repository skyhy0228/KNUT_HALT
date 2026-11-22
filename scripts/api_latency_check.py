"""제출 직전 운영진 LLM API 응답 속도 점검(개발용).

운영진 API가 혼잡하면 채점 중 LLM 호출이 시간 초과로 끝나 대체 답변이 나가고, 서버 쪽 문제로 낮게 나온 점수는
다시 채점하지 않는다(SPEC §6). 마지막 제출 직전에 이 스크립트로 API 속도를 보고, 느리면 나중에 제출한다.

- 서버와 같은 설정(.env의 OPENAI_*, 채팅 모델, 추론 수준)으로 짧은 JSON 호출을 --calls번(기본 4) 동시에 보낸다.
- 입력은 KB 섹션 한 개(kb/finance/FIN-03-02.md)와 고정 질문뿐이다. 채점 요청·공개 평가 데이터는 쓰지 않는다.
- 화면에는 호출별 소요 시간과 성공 여부만 출력한다. API 키·토큰·응답 본문은 출력하지 않는다.
- 헤지·재시도 없이 한 번씩만 부르므로, 서버가 실제로 겪는 호출 시간을 그대로 보여 준다.

사용: python3 scripts/api_latency_check.py [--calls 4] [--rounds 1] [--log /tmp/server8228_v33.19.log]
  --log: 서버 기록에서 최근 30분 동안의 LLM 시간 초과·오류 경고 수도 함께 센다(본문은 기록에 없음).
판정 기준(2026-11-22 20:15 KST, 채점이 없을 때 잰 값 기준): 최대 8초 미만이고 실패 0이면 '정상',
최대 15초 미만이면 '느림', 그 밖(실패가 있거나 15초 이상)은 '혼잡'.
"""
import argparse
import asyncio
import datetime as dt
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402  (.env를 읽는다)
from app.llm import LLM  # noqa: E402

PROMPT = ("너는 학습 자료를 읽고 짧게 답하는 도우미다. JSON 객체 하나로만 답한다: "
          '{"sentences": [자료에서 체크카드 사용 규칙을 말하는 문장 2개], "rule_count": 정수}')
QUESTION = "이 자료에서 체크카드 사용 규칙 두 가지를 찾아 줘."
NORMAL_MAX_S, SLOW_MAX_S = 8.0, 15.0


async def one_call(llm: LLM, messages, timeout: float):
    t0 = time.perf_counter()
    try:
        await llm._json_once(messages, timeout, llm.s.max_completion_tokens)
        return time.perf_counter() - t0, "ok"
    except asyncio.TimeoutError:
        return time.perf_counter() - t0, "timeout"
    except Exception as e:  # 네트워크·API·JSON 오류: 종류만
        return time.perf_counter() - t0, type(e).__name__


def recent_warnings(log_path: Path, minutes: int = 30) -> dict:
    """서버 기록의 최근 minutes분 경고 수(llm_timeout·llm_error·llm_json_decode_error·llm_hedged)."""
    since = dt.datetime.now() - dt.timedelta(minutes=minutes)
    counts = {k: 0 for k in ("llm_timeout", "llm_error", "llm_json_decode_error", "llm_hedged")}
    reqs = 0
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    for line in log_path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if not m or dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < since:
            continue
        if " req path=/" in line:
            reqs += 1
        for k in counts:
            if f"WARNING {k}" in line:
                counts[k] += 1
    counts["requests"] = reqs
    return counts


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=4, help="한 번에 동시에 보낼 호출 수")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--log", default=None, help="서버 기록 파일(선택)")
    a = ap.parse_args()

    settings = load_settings()
    llm = LLM(settings)
    section = (ROOT / "kb/finance/FIN-03-02.md").read_text(encoding="utf-8")
    messages = [{"role": "system", "content": PROMPT},
                {"role": "system", "content": "[자료]\n" + section},
                {"role": "user", "content": QUESTION}]
    print(f"모델 {settings.chat_model}, 추론 수준 {settings.reasoning_effort}, 호출 {a.calls}개 동시 × {a.rounds}회")
    times, fails = [], []
    try:
        for r in range(a.rounds):
            res = await asyncio.gather(*(one_call(llm, messages, a.timeout) for _ in range(a.calls)))
            print(f"  {r + 1}회차: " + ", ".join(f"{t:.1f}초" + ("" if s == "ok" else f"({s})") for t, s in res))
            times += [t for t, s in res if s == "ok"]
            fails += [s for _, s in res if s != "ok"]
    finally:
        await llm.close()
    if times:
        ts = sorted(times)
        print(f"성공 {len(times)}개: 중앙값 {ts[len(ts) // 2]:.1f}초, 최대 {ts[-1]:.1f}초 / 실패 {len(fails)}개")
    else:
        print(f"성공 0개 / 실패 {len(fails)}개")
    worst = max(times) if times else float("inf")
    if not fails and worst < NORMAL_MAX_S:
        verdict = "정상 — 지금 제출해도 됩니다"
    elif not fails and worst < SLOW_MAX_S:
        verdict = "느림 — 몇 분 뒤 다시 재 보고 제출하세요"
    else:
        verdict = "혼잡 — 지금 제출하면 시간 초과로 점수가 떨어질 수 있습니다"
    print("판정:", verdict)
    if a.log:
        p = Path(a.log)
        if p.exists():
            print("서버 기록 최근 30분:", recent_warnings(p))
        else:
            print("서버 기록 파일 없음:", a.log)


if __name__ == "__main__":
    asyncio.run(main())
