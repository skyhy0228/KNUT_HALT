"""오프라인 KB 가공 자료 만들기(개발용, 서버 시작 전에 한 번 실행).

규정(SPEC §7): 서버에 넣어 쓸 수 있는 것 = 배포 자료와 그것을 가공한 것(운영진 제공 LLM으로 만든 것 포함).
그래서 입력은 지식 베이스(kb/)뿐이고, 모델은 운영진 제공 채팅 모델(.env의 OPENAI_*)만 쓴다. 공개 세트·채점 요청은 쓰지 않는다.
추론 강도(--effort, 기본 medium)를 올리면 출력 토큰이 늘어 비스트리밍 상한(2000)에 걸리므로 스트리밍으로 받는다.

만드는 것(섹션마다 한 번씩 호출):
  aliases  → data/kb_aliases.json   C-8 별칭·영문명 사전(이름이 KB 글자 그대로 있는 것만)
  coverage → data/kb_coverage.json  섹션별 '다루는 것'(학생이 이 섹션만 보고 답을 찾을 수 있는 거리)과
                                     '다루지 않는 것'(같은 대상에 대해 물을 만하지만 섹션에 없는 거리), 대상이 KB에 있는 것만
  facts    → data/kb_facts.json     섹션별 사실 (대상, 속성, 값). 값이 그 섹션 본문에 글자 그대로 있는 것만
사용: python3 scripts/build_kb_offline.py [--task all|aliases|coverage|facts] [--effort medium] [--concurrency 6]
"""
import argparse
import asyncio
import datetime
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALIAS_PROMPT = (
    "[학습 자료]에서 이 과정이 지어낸 고유한 이름을 빠짐없이 찾는다: 가상 은행·통장·카드·적금·예금·포인트·앱·서비스·도구·"
    "학교·동아리·인물·실습·예제·활동·규칙의 이름. 통장·이자·리스트처럼 흔한 일반 명사는 넣지 않는다.\n"
    "각 이름마다, 영어로 질문하는 학습자가 같은 대상을 부를 때 쓸 만한 영어 표현을 1~3개 적는다. "
    "뜻을 옮긴 영어 이름(예: '토끼 적금' → 'Rabbit Savings')과 로마자 표기(예: 'Tokki Savings')를 함께 쓴다. "
    "사람 이름 같은 한 글자 이름은 로마자로 적는다(예: '학생 나' → 'student Na').\n"
    "한국어 이름은 [학습 자료]에 적힌 글자 그대로 적는다. JSON 객체 하나로만 답한다: "
    '{"names": [{"ko": "자료 속 이름", "en": ["영어 표현", "..."]}]}'
)
COVERAGE_PROMPT = (
    "[학습 자료] 한 섹션을 읽고 두 목록을 만든다.\n"
    "1) covers: 학생이 이 섹션만 보고 답을 찾을 수 있는 질문 거리를 '대상 — 물을 거리' 꼴로 빠짐없이 적는다. "
    "대상은 자료에 적힌 이름 그대로(실습·예제·상품·도구·인물·개념 이름). 물을 거리는 학생이 물을 만한 속성·규칙·값·절차·결과를 짧게.\n"
    "2) not_covered: 같은 대상들에 대해 학생이 물을 만하지만 이 섹션에 답이 적혀 있지 않은 거리를 같은 꼴로 적는다. "
    "섹션에 답이 있는 것은 넣지 않는다. 일반 상식이 아니라 이 과정 고유의 규칙·기준·수치·기능 위주로.\n"
    "자료에 없는 내용을 covers에 넣지 않는다. JSON 객체 하나로만 답한다: "
    '{"covers": ["대상 — 물을 거리", "..."], "not_covered": ["대상 — 물을 거리", "..."]}'
)
FACTS_PROMPT = (
    "[학습 자료] 한 섹션에 적힌 사실을 빠짐없이 (대상, 속성, 값) 꼴로 적는다. 표의 칸·목록·코드 예제의 결과도 사실로 적는다.\n"
    "- s(대상): 자료에 적힌 이름 그대로. p(속성): 무엇에 대한 값인지 짧게. v(값): 자료에 적힌 글자 그대로 "
    "(숫자·단위·조건을 바꾸거나 계산하지 않는다).\n"
    "자료에 없는 사실은 넣지 않는다. JSON 객체 하나로만 답한다: "
    '{"facts": [{"s": "대상", "p": "속성", "v": "값"}]}'
)
_EN_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '\-.&]{1,58}[A-Za-z0-9.]$")


def _flat(t: str) -> str:
    from app.textnorm import normalize
    return re.sub(r"\s+", "", normalize(str(t)))


async def stream_json(client, model: str, messages, effort: str, max_tokens: int = 8000, timeout: float = 300.0):
    """스트리밍으로 받아 이어 붙인 JSON 객체. 비스트리밍 상한(2000)을 넘는 출력도 끊기지 않게 받는다."""
    stream = await client.chat.completions.create(
        model=model, messages=messages, stream=True, response_format={"type": "json_object"},
        max_completion_tokens=max_tokens, extra_body={"reasoning_effort": effort}, timeout=timeout)
    parts, finish = [], None
    async for chunk in stream:
        for ch in chunk.choices or []:
            if ch.delta is not None and ch.delta.content:
                parts.append(ch.delta.content)
            if ch.finish_reason:
                finish = ch.finish_reason
    if finish == "length":
        raise ValueError("출력이 한도에서 끊김")
    obj = json.loads("".join(parts))
    if not isinstance(obj, dict):
        raise ValueError("JSON 객체가 아님")
    return obj


async def run_task(task: str, kb, client, model: str, effort: str, concurrency: int):
    prompt = {"aliases": ALIAS_PROMPT, "coverage": COVERAGE_PROMPT, "facts": FACTS_PROMPT}[task]
    sem = asyncio.Semaphore(concurrency)
    failed = []

    async def one(sid):
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": f"[학습 자료] {sid}\n{kb[sid].text}"}]
        for attempt in range(3):
            async with sem:
                try:
                    return sid, await stream_json(client, model, msgs, effort)
                except Exception as e:  # noqa: BLE001 — 끊김·형식 오류·일시 오류는 다시 받는다
                    err = type(e).__name__
            await asyncio.sleep(2 + 3 * attempt)
        failed.append((sid, err))
        return sid, None
    t0 = time.time()
    results = await asyncio.gather(*[one(sid) for sid in kb])
    print(f"[{task}] 섹션 {len(kb)}개 응답 {sum(o is not None for _, o in results)}개, 실패 {failed}, {time.time() - t0:.0f}초", flush=True)
    return results, failed


def build_aliases(results, kb, kb_flat):
    merged, dropped = {}, 0
    for sid, obj in results:
        for n in (obj or {}).get("names") or []:
            ko = " ".join(str(n.get("ko", "")).split()) if isinstance(n, dict) else ""
            ens = n.get("en") if isinstance(n, dict) else None
            if not ko or len(ko) < 2 or _flat(ko) not in kb_flat or not isinstance(ens, list):
                dropped += 1
                continue
            good = [" ".join(e.split()) for e in ens if isinstance(e, str) and _EN_OK.match(" ".join(e.split()))]
            if not good:
                dropped += 1
                continue
            entry = merged.setdefault(ko, {"en": [], "sections": []})
            for e in good:
                if e.lower() not in {x.lower() for x in entry["en"]}:
                    entry["en"].append(e)
            if sid not in entry["sections"]:
                entry["sections"].append(sid)
    return {"aliases": dict(sorted(merged.items()))}, f"이름 {len(merged)}개, 버린 항목 {dropped}개"


def build_coverage(results, kb, kb_flat):
    out, kept, dropped = {}, 0, 0
    for sid, obj in results:
        if obj is None:
            continue
        sec_flat = _flat(kb[sid].text)
        entry = {"covers": [], "not_covered": []}
        for key in ("covers", "not_covered"):
            for it in obj.get(key) or []:
                if not isinstance(it, str) or "—" not in it and "-" not in it:
                    dropped += 1
                    continue
                subj, _, what = it.strip().partition("—" if "—" in it else "-")
                subj, what = " ".join(subj.split()), " ".join(what.split())
                # 대상은 다루는 것이면 그 섹션에, 다루지 않는 것이면 KB 어딘가에 글자 그대로 있어야 한다
                ok_subj = subj and (_flat(subj) in (sec_flat if key == "covers" else kb_flat))
                if not ok_subj or not what or len(what) > 60:
                    dropped += 1
                    continue
                entry[key].append({"s": subj, "q": what})
                kept += 1
        out[sid] = entry
    return {"coverage": out}, f"항목 {kept}개(다루는 것 {sum(len(v['covers']) for v in out.values())}·다루지 않는 것 {sum(len(v['not_covered']) for v in out.values())}), 버린 항목 {dropped}개"


def build_facts(results, kb, kb_flat):
    out, kept, dropped = {}, 0, 0
    for sid, obj in results:
        if obj is None:
            continue
        sec_flat = _flat(kb[sid].text)
        rows = []
        for f in obj.get("facts") or []:
            if not isinstance(f, dict):
                dropped += 1
                continue
            s, p, v = (" ".join(str(f.get(k, "")).split()) for k in ("s", "p", "v"))
            # 값은 그 섹션 본문에 글자 그대로, 대상은 KB 어딘가에 있어야 한다(지어낸 사실을 거른다)
            if not s or not p or not v or len(v) > 80 or _flat(v) not in sec_flat or _flat(s) not in kb_flat:
                dropped += 1
                continue
            rows.append({"s": s, "p": p, "v": v})
            kept += 1
        out[sid] = rows
    return {"facts": out}, f"사실 {kept}개, 버린 항목 {dropped}개"


async def main(a):
    from app.config import load_settings
    from app.kb import load_kb
    from openai import AsyncOpenAI
    settings = load_settings()
    kb = load_kb(settings.kb_dir)
    kb_flat = _flat(" ".join(s.text for s in kb.values()))
    client = AsyncOpenAI(max_retries=0)
    tasks = ["aliases", "coverage", "facts"] if a.task == "all" else [a.task]
    builders = {"aliases": build_aliases, "coverage": build_coverage, "facts": build_facts}
    files = {"aliases": "kb_aliases.json", "coverage": "kb_coverage.json", "facts": "kb_facts.json"}
    outs = await asyncio.gather(*[run_task(t, kb, client, settings.chat_model, a.effort, a.concurrency) for t in tasks])
    await client.close()
    for t, (results, failed) in zip(tasks, outs):
        data, summary = builders[t](results, kb, kb_flat)
        data["_about"] = {
            "made_with": f"운영진 제공 채팅 모델 {settings.chat_model}, 추론 강도 {a.effort}, 스트리밍(scripts/build_kb_offline.py)",
            "input": "kb/만(공개 세트·채점 요청은 쓰지 않음)",
            "made_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "failed_sections": [s for s, _ in failed],
        }
        path = Path(a.out_dir) / files[t]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[{t}] 저장 {path} — {summary}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "aliases", "coverage", "facts"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    asyncio.run(main(ap.parse_args()))
