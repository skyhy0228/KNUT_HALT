"""C-8 별칭·영문명 사전 만들기(개발용, 한 번 실행해 data/kb_aliases.json을 만든다).

규정(SPEC §7): 서버에 넣어 쓸 수 있는 것 = 배포 자료와 그것을 가공한 것(운영진 제공 LLM으로 만든 것 포함).
그래서 이 사전은 **지식 베이스(kb/)만 입력으로, 운영진 제공 채팅 모델(gpt-5-nano, .env의 OPENAI_*)로** 만든다.
공개 세트·채점 요청은 입력으로 쓰지 않는다. 다른 LLM이 쓴 목록을 넣지 않는다.

절차: 섹션(48개)마다 본문(md + 표)을 주고 '이 과정에서 지어낸 이름'과 영어 사용자가 쓸 영어 표현(뜻 번역·로마자)을 JSON으로 받는다
→ 한국어 이름이 KB 본문에 글자 그대로 있는 것만 남기고, 영어 표현은 영문·숫자·공백·하이픈만 남긴다 → 이름별로 합친다.
사용: python3 scripts/build_aliases.py [--out data/kb_aliases.json]
"""
import argparse
import asyncio
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = (
    "[학습 자료]에서 이 과정이 지어낸 고유한 이름을 빠짐없이 찾는다: 가상 은행·통장·카드·적금·예금·포인트·앱·서비스·도구·"
    "학교·동아리·인물·실습·예제·활동·규칙의 이름. 통장·이자·리스트처럼 흔한 일반 명사는 넣지 않는다.\n"
    "각 이름마다, 영어로 질문하는 학습자가 같은 대상을 부를 때 쓸 만한 영어 표현을 1~3개 적는다. "
    "뜻을 옮긴 영어 이름(예: '토끼 적금' → 'Rabbit Savings')과 로마자 표기(예: 'Tokki Savings')를 함께 쓴다. "
    "사람 이름 같은 한 글자 이름은 로마자로 적는다(예: '학생 나' → 'student Na').\n"
    "한국어 이름은 [학습 자료]에 적힌 글자 그대로 적는다. JSON 객체 하나로만 답한다: "
    '{"names": [{"ko": "자료 속 이름", "en": ["영어 표현", "..."]}]}'
)
_EN_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '\-.&]{1,58}[A-Za-z0-9.]$")


async def main(out_path: Path):
    from app.config import load_settings
    from app.kb import load_kb
    from app.llm import LLM
    settings = load_settings()
    kb = load_kb(settings.kb_dir)
    kb_text = " ".join(" ".join(s.text.split()) for s in kb.values())
    llm = LLM(settings)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(8)

    async def one(sid):
        async with sem:
            msgs = [{"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"[학습 자료] {sid}\n{kb[sid].text}"}]
            return sid, await llm.chat_json(msgs, loop.time() + 60, max_tokens=2000)
    results = await asyncio.gather(*[one(sid) for sid in kb])
    await llm.close()

    merged, dropped, failed = {}, 0, []
    for sid, obj in results:
        names = obj.get("names") if isinstance(obj, dict) else None
        if not isinstance(names, list):
            failed.append(sid)
            continue
        for n in names:
            ko = " ".join(str(n.get("ko", "")).split()) if isinstance(n, dict) else ""
            ens = n.get("en") if isinstance(n, dict) else None
            if not ko or ko not in kb_text or len(ko) < 2 or not isinstance(ens, list):
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
    data = {
        "_about": {
            "what": "C-8 별칭·영문명 사전: KB 속 고유한 이름 → 영어 표현",
            "made_with": f"운영진 제공 채팅 모델 {settings.chat_model}(scripts/build_aliases.py), 입력은 kb/만",
            "made_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "check": "한국어 이름이 KB 본문에 글자 그대로 있는 것만, 영어 표현은 영문·숫자·공백·하이픈만",
        },
        "aliases": dict(sorted(merged.items())),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"이름 {len(merged)}개 저장 → {out_path} | 버린 항목 {dropped}개 | 응답 실패 섹션 {failed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "kb_aliases.json"))
    a = ap.parse_args()
    asyncio.run(main(Path(a.out)))
