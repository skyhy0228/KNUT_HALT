"""오프라인 자료의 빠진 섹션만 다시 받기(개발용). API가 회복될 때까지 1분마다 확인하고, 회복되면 빠진 섹션만
build_kb_offline.py와 같은 프롬프트·거르기·추론 강도(medium)·스트리밍으로 받아 기존 파일에 합친다.
긴 섹션(다루는 것·사실 표)은 소제목(### ) 단위로 나눠 받아 출력이 짧게 한다. 사용: python3 scripts/offline_retry.py [--minutes 40]"""
import argparse, asyncio, json, re, sys, time, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import build_kb_offline as B  # noqa: E402

FILES = {"aliases": "kb_aliases.json", "coverage": "kb_coverage.json", "facts": "kb_facts.json"}
PROMPTS = {"aliases": B.ALIAS_PROMPT, "coverage": B.COVERAGE_PROMPT, "facts": B.FACTS_PROMPT}


def chunks(text: str, limit: int = 1500):
    parts, cur = [], ""
    for block in re.split(r"(?m)^(?=#{2,3} )", text):
        if cur and len(cur) + len(block) > limit:
            parts.append(cur); cur = ""
        cur += block
    if cur:
        parts.append(cur)
    return parts or [text]


def merge_objs(task, objs):
    if task == "aliases":
        return {"names": [n for o in objs for n in (o.get("names") or [])]}
    if task == "coverage":
        return {"covers": [x for o in objs for x in (o.get("covers") or [])],
                "not_covered": [x for o in objs for x in (o.get("not_covered") or [])]}
    return {"facts": [x for o in objs for x in (o.get("facts") or [])]}


def missing(task, data, kb):
    if task == "aliases":
        return list(dict.fromkeys(data.get("_about", {}).get("failed_sections", [])))
    key = "coverage" if task == "coverage" else "facts"
    have = data.get(key, {})
    return [sid for sid in kb if sid not in have]


async def healthy(client, model):
    try:
        r = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": 'JSON으로만: {"ok": true}'}],
                                                 max_completion_tokens=200, response_format={"type": "json_object"},
                                                 extra_body={"reasoning_effort": "low"}, timeout=30)
        return bool(r.choices[0].message.content)
    except Exception:  # noqa: BLE001
        return False


async def main(a):
    from app.config import load_settings
    from app.kb import load_kb
    from openai import AsyncOpenAI
    s = load_settings(); kb = load_kb(s.kb_dir); kb_flat = B._flat(" ".join(x.text for x in kb.values()))
    client = AsyncOpenAI(max_retries=0); sem = asyncio.Semaphore(a.concurrency)
    wanted = [t for t in a.tasks.split(",") if t in FILES]
    save_lock = asyncio.Lock()
    t_end = time.time() + a.minutes * 60
    builders = {"aliases": B.build_aliases, "coverage": B.build_coverage, "facts": B.build_facts}
    while time.time() < t_end:
        datas = {t: json.loads((ROOT / "data" / f).read_text(encoding="utf-8")) for t, f in FILES.items()}
        todo = {t: (missing(t, datas[t], kb) if t in wanted else []) for t in FILES}
        n = sum(len(v) for v in todo.values())
        if n == 0:
            print(f"{dt.datetime.utcnow():%H:%M:%S} 빠진 섹션 없음 — 끝", flush=True); break
        if not await healthy(client, s.chat_model):
            print(f"{dt.datetime.utcnow():%H:%M:%S} API 아직 불가, 빠진 섹션 {n}개", flush=True)
            await asyncio.sleep(60); continue
        print(f"{dt.datetime.utcnow():%H:%M:%S} API 정상 — 빠진 섹션 {n}개 다시 받기 " + str({t: len(v) for t, v in todo.items()}), flush=True)

        async def save(task, sid, obj):
            """섹션 하나를 받는 즉시 파일을 다시 읽어 합치고 저장한다(중간에 끊겨도 받은 섹션은 남는다)."""
            async with save_lock:
                path = ROOT / "data" / FILES[task]
                data = json.loads(path.read_text(encoding="utf-8"))
                new, _ = builders[task]([(sid, obj)], kb, kb_flat)
                if task == "aliases":
                    for ko, v in new["aliases"].items():
                        e = data["aliases"].setdefault(ko, {"en": [], "sections": []})
                        e["en"] += [x for x in v["en"] if x.lower() not in {y.lower() for y in e["en"]}]
                        e["sections"] += [x for x in v["sections"] if x not in e["sections"]]
                    data["aliases"] = dict(sorted(data["aliases"].items()))
                    data["_about"]["failed_sections"] = [x for x in data["_about"].get("failed_sections", []) if x != sid]
                else:
                    key = "coverage" if task == "coverage" else "facts"
                    data[key].update(new[key])
                data["_about"]["retried_at"] = (dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                                                + f" (빠진 섹션만, {a.chunk}자 조각, 출력 한도 {a.max_tokens})")
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                tmp.replace(path)  # 쓰는 도중에 읽혀도 깨진 파일이 보이지 않게 바꿔치기
            print(f"   {dt.datetime.utcnow():%H:%M:%S} {task} {sid} 저장", flush=True)

        async def one(task, sid):
            pieces = [kb[sid].text] if task == "aliases" else chunks(kb[sid].text, a.chunk)
            objs = []
            for j, piece in enumerate(pieces):
                if time.time() > t_end:
                    return sid, None, "시간 한도"
                msgs = [{"role": "system", "content": PROMPTS[task]}, {"role": "user", "content": f"[학습 자료] {sid}\n{piece}"}]
                got = None
                for attempt in range(3):
                    async with sem:
                        t0 = time.time()
                        try:
                            got = await B.stream_json(client, s.chat_model, msgs, a.effort, max_tokens=a.max_tokens)
                            print(f"   {dt.datetime.utcnow():%H:%M:%S} {task} {sid} 조각 {j + 1}/{len(pieces)} 성공 {time.time() - t0:.0f}초", flush=True)
                            break
                        except Exception as e:  # noqa: BLE001
                            print(f"   {dt.datetime.utcnow():%H:%M:%S} {task} {sid} 조각 {j + 1}/{len(pieces)} 실패 {time.time() - t0:.0f}초 {type(e).__name__}", flush=True)
                    await asyncio.sleep(5 + 10 * attempt)
                if got is None:
                    return sid, None, "조각 실패"
                objs.append(got)
            obj = merge_objs(task, objs)
            await save(task, sid, obj)
            return sid, obj, None
        for task, sids in todo.items():
            if not sids:
                continue
            res = await asyncio.gather(*[one(task, sid) for sid in sids])
            print(f"   [{task}] 이번 바퀴 저장 {sum(o is not None for _, o, _ in res)}개, "
                  f"아직 빠짐 {sum(o is None for _, o, _ in res)}개", flush=True)
        await asyncio.sleep(30)  # 바퀴 사이에 쉬어 불안정한 API를 몰아치지 않는다
    await client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=int, default=40); ap.add_argument("--effort", default="medium")
    ap.add_argument("--tasks", default="aliases,coverage,facts"); ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=12000)  # 추론 토큰 포함. 너무 크면 추론 중 무응답이 길어져 중간 서버 시간 초과(504) 위험
    asyncio.run(main(ap.parse_args()))
