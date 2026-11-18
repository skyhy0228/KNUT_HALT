"""공개 세트(배포 자료)에서 질문과 비슷한 예시를 찾아 few-shot으로 쓴다(SPEC §7.1 허용).

- 예시 풀은 배포 자료 파일에서만 만든다(FEWSHOT_FILES). 채점 요청은 풀에 넣지 않는다.
- 마지막 발화가 정규화 후 완전히 같은 예시는 건너뛴다(공개 세트로 로컬 평가할 때 자기 자신이 예시가 되는 누수 방지).
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .retrieval import BM25
from .textnorm import normalize, tokenize

log = logging.getLogger("halt")
_CITE_TAIL = re.compile(r"\s*\((?:PY|PR|FIN)-\d{2}-\d{2}(?:,\s*(?:PY|PR|FIN)-\d{2}-\d{2})*\)\s*$")


@dataclass
class Example:
    endpoint: str
    last: str
    prev_user: str
    category: str
    action: Optional[str]
    reference: Optional[str]
    has_candidate: bool
    candidate: Optional[str]
    grounded: Optional[bool]
    safe: Optional[bool]
    unsupported: List[str]


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# EXAMPLE_FMT=v2(기본): 예시 표시 정리. v1은 v14 그대로.
# - 긴 발화는 앞 1/3과 뒤를 남기고 가운데를 줄인다. 앞만 남기면 요청의 핵심(대개 끝부분)이 잘려
#   서로 다른 문항이 같은 문장처럼 보이고, 같은 문장에 반대 라벨이 붙은 예시가 된다(v14 프롬프트에서 확인).
# - 줄인 발화가 이미 보여 준 예시와 같으면 그 예시는 빼고, action 라벨이 없으면(가드 세트 예시) "action: None"을 쓰지 않는다.
EXAMPLE_FMT = os.environ.get("EXAMPLE_FMT", "v2")


def _clip2(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    head, tail = s[:n // 3], s[-(n - n // 3 - 3):]
    if " " in head:
        head = head[:head.rfind(" ")]      # 단어 중간에서 끊지 않는다
    if " " in tail:
        tail = tail[tail.find(" ") + 1:]
    return head.rstrip() + " … " + tail.lstrip()


def _fmt_last(e: "Example", n: int) -> str:
    return _clip2(e.last, n) if EXAMPLE_FMT == "v2" else _clip(e.last, n)


def _dedupe(exs: List["Example"], n: int) -> List["Example"]:
    if EXAMPLE_FMT != "v2":
        return exs
    seen, out = set(), []
    for e in exs:
        k = _fmt_last(e, n)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


class ExamplePool:
    def __init__(self, files: Sequence[Path]):
        self.items: List[Example] = []
        for f in files:
            if not Path(f).exists():
                log.warning("fewshot_file_missing")
                continue
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                p = json.loads(line)
                lab = p.get("labels") or {}
                msgs = p.get("messages") or []
                users = [m["content"] for m in msgs if m.get("role") == "user"]
                if not users:
                    continue
                ref = lab.get("reference_answer")
                self.items.append(Example(
                    endpoint=p.get("endpoint", ""), last=users[-1],
                    prev_user=users[-2] if len(users) > 1 else "",
                    category=lab.get("category", "normal"), action=lab.get("action"),
                    reference=_CITE_TAIL.sub("", ref) if isinstance(ref, str) else None,
                    has_candidate="candidate_answer" in p, candidate=p.get("candidate_answer"),
                    grounded=lab.get("grounded"), safe=lab.get("safe"),
                    unsupported=list(lab.get("unsupported") or [])))
        self._norm = [normalize(e.last).strip() for e in self.items]
        self.bm25 = BM25([tokenize(e.last + " " + e.prev_user) for e in self.items]) if self.items else None
        log.info("fewshot_pool size=%d", len(self.items))

    def search(self, messages: List[Dict[str, str]], k: int, endpoint: Optional[str] = None,
               with_candidate: Optional[bool] = None, dedupe: int = 0,
               exclude_unsafe: bool = False) -> List[Example]:
        """dedupe=n(EXAMPLE_FMT=v2일 때만): n자로 줄여 보인 발화가 앞 예시와 같으면 건너뛰고 다음 순위로 k개를 채운다.
        exclude_unsafe: 안전하지 않은 후보 답변 예시(safe=False)를 뺀다(가드가 safe를 LLM에 묻지 않을 때)."""
        if not self.items or not messages:
            return []
        seen = set()
        last = messages[-1]["content"]
        users = [m["content"] for m in messages if m["role"] == "user"]
        prev = users[-2] if len(users) > 1 else ""
        sc = self.bm25.scores(tokenize(last + " " + last + " " + prev))
        me = normalize(last).strip()
        out: List[Example] = []
        for i in np.argsort(-sc):
            e = self.items[i]
            if sc[i] <= 0 or len(out) >= k:
                break
            if self._norm[i] == me:
                continue
            if endpoint and e.endpoint != endpoint:
                continue
            if with_candidate is not None and e.has_candidate != with_candidate:
                continue
            if exclude_unsafe and e.safe is False:
                continue
            if dedupe and EXAMPLE_FMT == "v2":
                key = _fmt_last(e, dedupe)
                if key in seen:
                    continue
                seen.add(key)
            out.append(e)
        return out


def format_tutor_examples(exs: List[Example]) -> str:
    lines = []
    for e in _dedupe(exs, 160):
        label = f"범주: {e.category}" + (f", action: {e.action}" if e.action or EXAMPLE_FMT != "v2" else "")
        lines.append(f"- 학습자: {_fmt_last(e, 160)}\n  → {label}"
                     + (f"\n  → 모범 답변: {_clip(e.reference, 160)}" if e.reference else ""))
    return "\n".join(lines)


def format_guard_input_examples(exs: List[Example]) -> str:
    return "\n".join(f"- 학습자: {_fmt_last(e, 160)}\n  → 범주: {e.category}" for e in _dedupe(exs, 160))


def format_guard_output_examples(exs: List[Example], with_safe: bool = True) -> str:
    """with_safe=False: 판정 예시에서 safe를 뺀다(가드가 safe를 입력 판정으로 정해 LLM에 묻지 않을 때)."""
    lines = []
    for e in exs:
        verdict = ({"safe": e.safe} if with_safe else {}) | {"grounded": e.grounded, "unsupported": e.unsupported}
        lines.append(f"- 학습자: {_fmt_last(e, 120)}\n  후보 답변: {_clip(e.candidate or '', 200)}\n"
                     f"  → 판정: {json.dumps(verdict, ensure_ascii=False)}")
    return "\n".join(lines)
