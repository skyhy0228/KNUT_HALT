"""공개 세트(배포 자료)에서 질문과 비슷한 예시를 찾아 few-shot으로 쓴다(SPEC §7.1 허용).

- 예시 풀은 배포 자료 파일에서만 만든다(FEWSHOT_FILES). 채점 요청은 풀에 넣지 않는다.
- 마지막 발화가 정규화 후 완전히 같은 예시는 건너뛴다(공개 세트로 로컬 평가할 때 자기 자신이 예시가 되는 누수 방지).
"""
import json
import logging
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
               with_candidate: Optional[bool] = None) -> List[Example]:
        if not self.items or not messages:
            return []
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
            out.append(e)
        return out


def format_tutor_examples(exs: List[Example]) -> str:
    lines = []
    for e in exs:
        lines.append(f"- 학습자: {_clip(e.last, 160)}\n  → 범주: {e.category}, action: {e.action}"
                     + (f"\n  → 모범 답변: {_clip(e.reference, 160)}" if e.reference else ""))
    return "\n".join(lines)


def format_guard_input_examples(exs: List[Example]) -> str:
    return "\n".join(f"- 학습자: {_clip(e.last, 160)}\n  → 범주: {e.category}" for e in exs)


def format_guard_output_examples(exs: List[Example]) -> str:
    lines = []
    for e in exs:
        verdict = {"safe": e.safe, "grounded": e.grounded, "unsupported": e.unsupported}
        lines.append(f"- 학습자: {_clip(e.last, 120)}\n  후보 답변: {_clip(e.candidate or '', 200)}\n"
                     f"  → 판정: {json.dumps(verdict, ensure_ascii=False)}")
    return "\n".join(lines)
