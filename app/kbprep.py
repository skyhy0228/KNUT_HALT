"""KB 전처리(KBPREP, v23~). 원본 kb/는 그대로 두고, 서버 시작 때 파생 색인을 만든다(색인 생성에는 LLM을 쓰지 않는다).
C-8 별칭 사전만 운영진 제공 LLM으로 미리 만든 KB 가공물(data/kb_aliases.json)을 읽는다.

기본 켬(v26):
  C-8 별칭·영문명 사전: data/kb_aliases.json(scripts/build_aliases.py가 kb/만 입력으로 운영진 제공 gpt-5-nano로 만듦,
                       SPEC §7 '배포 자료를 가공한 것'). 영어 질문에서 영어 이름을 찾아 한국어 이름을 검색 질의와 번역 힌트로 준다.
  E-11·E-12 사실 단위 : 문장 ID(섹션#소제목.번호)와 숫자+단위 값을 모아, 가드가 '근거 없음' 후보의 틀린 값 구간을 짧게 잡는다.
기본 끔(실제 채점·공개 세트에서 해가 확인됨, _DEFAULTS 주석 참고) — 스위치 KBPREP_<요소>=1로만 켠다:
  A-1 2·3위 섹션은 질문과 맞는 소제목 청크만 [자료]에 / A-2 FAQ 질문 색인 / A-3 표 문장화 / B-4 공통 이름 순위 /
  B-5 같은 사실 인용 바꾸기 / B-6 상투 문구 빼기 / C-7 숫자 정규화 토큰 / C-9 띄어쓰기 합치기 / D-10·F-14 자료 밖 보조 판단 / F-13 코드 색인.
  기본값이면 영어 질문(별칭이 걸릴 때)을 빼고 튜터·가드 검색과 튜터 동작이 v22와 같다.
"""
import csv
import io
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .kb import Section, _split_h2
from .textnorm import normalize, tokenize

log = logging.getLogger("halt")
KBPREP = os.environ.get("KBPREP", "1") == "1"


# 하위 요소 기본값: 공개 세트에서 해가 확인된 요소는 끈다(2026-11-20 측정).
#  B4 공통 이름 순위: 인용 우선 순위에 넣으면 1위 적중 107→99(조건부로 써도 고침 1·깨짐 2).
#  B5 같은 사실 인용 바꾸기: 저장 출력 600건 재생에서 바뀐 1건이 인용 F1 -1.0.
#  D10 자료 밖 보조 판단: 재생에서 기준 0.5/0.6/0.7 모두 action 깨짐이 고침보다 많음(깨짐 1/2/3, 고침 0/1/1).
#  A1 2·3위 섹션 소제목만 넣기: 공개 세트 교차 검증에서는 해가 없었지만(action 173→176) 실제 채점에서 튜터가
#     v19~v22 10회 평균 87.18(85.47~88.72)에서 v23 2회 85.06·85.06으로 떨어졌다. 튜터 프롬프트의 93%를 바꾸는 요소라 v24에서 끈다.
#  v25: 튜터 쪽 전처리를 모두 끈다. 실제 채점 튜터 평균이 v19~v22 10회 87.18 → v23 4회 85.83 → v24(A-1 끔) 4회 86.34로,
#     검색 전처리(A-3·B-6·C-7·C-9)·A-2·F-13이 남은 v24도 0.84 낮았다. 가드 E-11(틀린 값 구간)만 켠다
#     (v23·v24 가드 8회 평균 91.01, v19~v22 90.65).
#  v26: C-8 별칭 사전을 켠다(영어 질문에서만 동작). 기본값이면 영어 질문을 빼고 튜터·가드 검색과 튜터 동작은 v22와 같다.
_DEFAULTS = {"B4": "0", "B5": "0", "D10": "0", "A1": "0", "A2": "0", "A3": "0", "B6": "0", "C7": "0", "C9": "0", "F13": "0",
             # v31 오프라인 자료(scripts/build_kb_offline.py, 운영진 제공 LLM medium·스트리밍, 입력은 kb/만)
             # 공개 재생: COV 고침 0·깨짐 0, FACT(구간 좁히기) 가드 +0.01·깨짐 0 → 켬.
             # FACTFLIP(근거 있음 → 없음 뒤집기)은 계산으로 나온 맞는 값(예: 한 달 이자)을 어긋난 값으로 볼 위험이 있어 끔.
             "COV": "1", "FACT": "1", "FACTFLIP": "0"}


def _on(name: str) -> bool:
    return KBPREP and os.environ.get(f"KBPREP_{name}", _DEFAULTS.get(name, "1")) == "1"


# ---------------- C-7 숫자·단위 정규화 ----------------
# 숫자는 앞이 숫자·점·쉼표가 아닌 곳에서 시작하고 자릿수에 상한을 둔다(긴 숫자열에서 정규식이 느려지거나
# float 변환이 넘치던 문제 — 코드 검토 2026-11-20).
_MONEY = re.compile(r"(?<![\d.,])(?:(\d{1,9}(?:\.\d{1,4})?)\s*만\s*(?:(\d{1,4})\s*천\s*)?|(\d{1,4})\s*천\s*|(\d{1,3}(?:,\d{3}){1,4}|\d{1,12})\s*)(?:원|won\b)", re.I)
_PCT = re.compile(r"(?<![\d.,])(?:(\d{1,3})\s*점\s*(\d{1,3})\s*(?:%p?|퍼센트(?:\s*포인트)?)|(\d{1,6}(?:\.\d{1,4})?)\s*(?:%p?|퍼센트(?:\s*포인트)?|percent))", re.I)
_AGE_RANGE = re.compile(r"만\s*(\d{1,2})\s*(?:세)?\s*(?:부터|[~∼\-])\s*(?:만\s*)?(\d{1,2})\s*세")
_AGE = re.compile(r"(?:만\s*)?(\d{1,2})\s*(?:세|살)(?![금트대계])")
_HOUR = re.compile(r"(오전|오후|밤|낮|저녁|새벽|아침)?\s*(\d{1,2})\s*시(?![간작험각키도대켜점])")


def _num(v: str) -> str:
    try:
        f = float(v)
        s = str(int(f)) if f == int(f) else ("%g" % f)
    except (OverflowError, ValueError):
        s = v
    return s.replace(".", "p")


def num_tokens(text: str) -> List[str]:
    """금액·비율·나이·시각을 같은 표기로 바꾼 토큰. 예: '2만 5천 원'·'25,000원'·'25000 won' → won25000."""
    t = str(text)
    out = []
    for m in _MONEY.finditer(t):
        a, b, c, d = m.groups()
        if a:
            v = int(round(float(a) * 10000)) + (int(b) * 1000 if b else 0)
        elif c:
            v = int(c) * 1000
        else:
            v = int(d.replace(",", ""))
        out.append(f"won{v}")
    for m in _PCT.finditer(t):
        x, y, z = m.groups()
        out.append("pct" + _num(f"{x}.{y}" if x else z))
    for m in _AGE_RANGE.finditer(t):
        out += [f"age{int(m.group(1))}", f"age{int(m.group(2))}"]
    for m in _AGE.finditer(t):
        out.append(f"age{int(m.group(1))}")
    for m in _HOUR.finditer(t):
        h = int(m.group(2))
        if m.group(1) in ("오후", "밤", "저녁") and h < 12:
            h += 12
        if h <= 24:
            out.append(f"h{h}")
    return out


# ---------------- A-3 표 문장화 ----------------
def table_sentences(csv_text: str) -> List[str]:
    rows = [r for r in csv.reader(io.StringIO(csv_text.strip())) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return []
    head, out = [h.strip() for h in rows[0]], []
    for r in rows[1:]:
        r = [c.strip() for c in r]
        if len(head) == 2 and len(r) >= 2:
            out.append(f"{head[1]} — {r[0]}: {r[1]}")
        else:
            pairs = [f"{h}: {v}" for h, v in zip(head[1:], r[1:]) if v and v not in ("—", "-", "–")]
            out.append(f"{head[0]} {r[0]} — " + ", ".join(pairs))
    return out


# ---------------- B-6 상투 문구 ----------------
_DISCLAIMER = re.compile(r"가상(?:의)?\s*은행|실제\s*은행")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------- F-13 코드 색인 ----------------
_PY_STOP = set("""and as assert break class continue def del elif else except false finally for from global if import in is
lambda none nonlocal not or pass raise return true try while with yield print input int str float list dict set tuple len range
sorted sum min max abs round type bool append get items keys values split join strip format enumerate zip open self""".split())
_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRLIT = re.compile(r"'([^'\n]{1,40})'|\"([^\"\n]{1,40})\"")


def code_terms(text: str) -> List[str]:
    """코드 블록(또는 코드처럼 보이는 질의)에서 식별자·문자열 상수를 뽑는다."""
    blocks = _FENCE.findall(text) or [text]
    out = []
    for b in blocks:
        out += [w.lower() for w in _IDENT.findall(b) if w.lower() not in _PY_STOP and len(w) >= 2]
        out += [x or y for x, y in _STRLIT.findall(b)]
    return out


# ---------------- 영어 질문 판별(튜터 v19 번역 검색·C-8 공용) ----------------
ENG_RATIO_MIN, ENG_MIN_LETTERS = 0.9, 8


def is_english(text: str) -> bool:
    """코드 블록이 없고 글자의 90% 이상이 로마자인 글."""
    if "```" in text:
        return False
    letters = re.findall(r"[A-Za-z가-힣]", text)
    return len(letters) >= ENG_MIN_LETTERS and sum(c.isascii() for c in letters) / len(letters) >= ENG_RATIO_MIN


# ---------------- E-11 숫자+단위 값 ----------------
_VALUE = re.compile(r"(?<![\w.,])(-?\d{1,12}(?:[.,]\d{1,6})?)(?![\d.,]\d)\s*(%p?|퍼센트|[가-힣]{1,2})?")
_PARTICLE_TAIL = re.compile(r"(?:이므로|이라서|이라고|이에요|예요|이며|이고|이|가|을|를|은|는|에|의|와|과|도|만)$")


_SID_REF = re.compile(r"[\[(]?\b(?:PY|PR|FIN)-\d{2}-\d{2}\b[\])]?")


def value_keys(text: str) -> Set[str]:
    """숫자+단위 꼴의 값 묶음(정규화 토큰 포함). 예: '연 1.3%' → {'1.3%', 'pct1p3'}, '1씩' → {'1씩'}."""
    text = _SID_REF.sub(" ", str(text))
    keys = set(num_tokens(text))
    for m in _VALUE.finditer(text):
        num, unit = m.group(1).replace(",", ""), (m.group(2) or "")
        unit = _PARTICLE_TAIL.sub("", unit) if len(unit) > 1 else unit
        keys.add(num + unit)
    return keys


@dataclass
class Unit:
    uid: str          # E-12 문장 ID: 섹션#소제목번호.번호
    sid: str
    h2: int
    kind: str         # sent | q | a | code | table
    text: str


def _grams(s: str, n: int = 4) -> Set[str]:
    s = re.sub(r"[\s\W_]+", "", normalize(s))
    return {s[i:i + n] for i in range(len(s) - n + 1)}


# v31 2-가 기준: 물을 거리의 글자 2-gram 중 묻는 항목에 들어 있어야 하는 비율
COV_MIN = float(os.environ.get("KBPREP_COV_MIN", "0.6"))
_GEN_UNITS = {"개월", "달", "주", "일", "년", "회", "번", "개", "명", "점", "분", "초", "층", "등", "위", "칸", "줄", "장", "권", "배", "단계"}


def _flat_text(t) -> str:
    return re.sub(r"[\s\W_]+", "", normalize(str(t)))


def _bigrams(t) -> set:
    f = _flat_text(t)
    return {f[i:i + 2] for i in range(len(f) - 1)}


def _clause_values(text: str) -> List[Tuple[int, int, str, str]]:
    """글 속 값의 (시작, 끝, 종류, 정규화 값). 종류: won·pct·age·h(C-7 정규화), u:단위(개월·회 등)."""
    vals, taken = [], []
    for rx, cls in ((_MONEY, "won"), (_PCT, "pct"), (_AGE_RANGE, "age"), (_AGE, "age"), (_HOUR, "h")):
        for m in rx.finditer(text):
            if any(s0 < m.end() and m.start() < e0 for s0, e0 in taken):
                continue
            for tok in num_tokens(m.group(0)):
                if tok.startswith(cls):
                    vals.append((m.start(), m.end(), cls, tok[len(cls):]))
            taken.append((m.start(), m.end()))
    for m in _VALUE.finditer(text):
        if any(s0 < m.end() and m.start() < e0 for s0, e0 in taken):
            continue
        unit = m.group(2) or ""
        unit = _PARTICLE_TAIL.sub("", unit) if len(unit) > 1 else unit
        if unit in _GEN_UNITS:
            vals.append((m.start(), m.end(), "u:" + unit, _num(m.group(1).replace(",", ""))))
    return vals


def _class_values(text: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for _, _, cls, val in _clause_values(str(text)):
        out.setdefault(cls, set()).add(val)
    return out


def _value_phrase(clause: str, start: int, end: int) -> str:
    """값 구절: 앞 수식어 두 낱말까지 + 값(뒤 단위·조사는 떼어 냄). E-11 구간과 같은 꼴."""
    left = clause[:start].rstrip().split(" ")
    lead = []
    for w in reversed(left[-2:]):
        if not w or re.search(r"(?:[.!?]|므로|면|고|서|며|해)$", w):
            break
        lead.insert(0, w)
    tail = re.match(r"[%가-힣]*", clause[end:])
    word = clause[start:end] + (tail.group(0) if tail else "")
    word = _PARTICLE_TAIL.sub("", word.strip()) or word.strip()
    return " ".join(lead + [word]).strip()


class KBPrep:
    def __init__(self, sections: Dict[str, Section], kb_dir=None):
        from .retrieval import BM25  # 순환 import 방지
        self.sections = sections
        self.kb_dir = kb_dir  # 서버가 load_kb에 쓴 폴더(없으면 KB_DIR 환경변수 또는 기본 kb/)
        self.section_bm25 = None  # Retriever가 만든 섹션 BM25를 연결한다(B-5 residual_rank에서 씀)
        self.section_ids = list(sections)
        self.tables: Dict[str, Dict[str, str]] = {}
        self.units: Dict[str, List[Unit]] = {}
        self._kb_text = "\n".join(normalize(s.text) for s in sections.values())
        disclaimer_home = None
        for sid in self.section_ids:
            if _DISCLAIMER.search(sections[sid].text):
                disclaimer_home = disclaimer_home or sid
        self.disclaimer_home = disclaimer_home

        sec_index, chunk_index = {}, []
        for sid in self.section_ids:
            sec = sections[sid]
            md_tables = self._section_tables(sid)
            self.tables[sid] = md_tables
            parts = _split_h2(self._md(sid))
            assert len(parts) == len(sec.chunks), f"청크 수 불일치: {sid}"
            units = []
            for hi, (heading, body) in enumerate(parts):
                units += self._units(sid, hi, body, md_tables)
            self.units[sid] = units
            # 색인 글 = 원래 글(v22와 같은 형식) + 전처리로 덧붙이는 부분. 모든 전처리를 끄면 v22 색인과 같다.
            sents = {n: table_sentences(t) if _on("A3") else [] for n, t in md_tables.items()}
            for c in sec.chunks:
                extra = [x for n, t in md_tables.items() if t in c.text for x in sents[n]]
                chunk_index.append(self._strip_disclaimer(sid, (c.heading or "") + "\n" + c.text) + ("\n" + "\n".join(extra) if extra else ""))
            sec_extra = [x for n in md_tables for x in sents[n]]
            sec_index[sid] = self._strip_disclaimer(sid, sec.text) + ("\n" + "\n".join(sec_extra) if sec_extra else "")
        self.section_index_text = sec_index           # 섹션 BM25용(B-6·A-3·C-7)
        self.chunk_index_text = chunk_index            # 청크 BM25용(Retriever.chunks 순서와 같음)

        # A-2 FAQ, F-13 코드
        self.faq = [(u.sid, f"{sections[u.sid].title} {u.text}") for sid in self.section_ids for u in self.units[sid] if u.kind == "q"]
        self.faq_bm25 = BM25([self._toks(t) for _, t in self.faq]) if self.faq else None
        self.code_vocab: Set[str] = set()
        code_docs = []
        for sid in self.section_ids:
            terms = code_terms("\n".join(u.text for u in self.units[sid] if u.kind == "code") or "")
            terms = [t for t in terms if t]
            self.code_vocab |= set(terms)
            code_docs.append([t.lower() for t in terms])
        self.code_bm25 = BM25(code_docs)

        # B-4 여러 섹션에 나오는 낱말 묶음(2~3어절) → 그 묶음이 든 청크 번호
        self.chunk_sid = [c.section_id for s in self.section_ids for c in sections[s].chunks]
        self.chunk_bm25 = None  # Retriever가 만든 청크 BM25를 연결한다
        ng_chunks: Dict[str, Set[int]] = defaultdict(set)
        word_df: Dict[str, Set[int]] = defaultdict(set)
        chunk_words = [re.findall(r"[가-힣a-z0-9]+", normalize(t)) for t in chunk_index]
        for ci, words in enumerate(chunk_words):
            for w in words:
                word_df[w].add(ci)
        rare_max = max(3, len(chunk_index) // 8)  # 청크의 1/8 넘게 나오는 낱말(조사·흔한 말)은 이름으로 보지 않는다
        for ci, words in enumerate(chunk_words):
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    ws = words[i:i + n]
                    if all(len(w) >= 2 and len(word_df[w]) <= rare_max for w in ws):
                        ng_chunks[" ".join(ws)].add(ci)
        self.name_chunks = {g: cs for g, cs in ng_chunks.items() if 2 <= len({self.chunk_sid[c] for c in cs}) <= 8}

        # B-5 같은 사실: 섹션별 문장 4-gram
        self.sent_grams = [(u.sid, u.uid, _grams(u.text)) for sid in self.section_ids for u in self.units[sid]
                           if u.kind in ("sent", "a", "table") and len(u.text) >= 15]

        self._load_aliases()  # C-8

        # E-11 섹션별 값 묶음
        self.section_values = {sid: set().union(*[value_keys(u.text) for u in self.units[sid]] or [set()])
                               for sid in self.section_ids}

    # ---- C-8 별칭·영문명 사전 ----
    _EN_STOP = {"the", "a", "an", "of", "for", "by", "at", "in", "and", "to", "my", "our", "your"}
    # 이것 하나만으로는 이름이라 볼 수 없는 흔한 영어 낱말(한 낱말짜리 표현은 이 낱말이면 버린다)
    _EN_GENERIC = {"bank", "card", "check", "debit", "saving", "savings", "point", "points", "reward", "rewards", "account",
                   "club", "school", "coding", "student", "teen", "teens", "tool", "agent", "assistant", "model", "service",
                   "program", "activity", "rule", "rules", "guide", "example", "practice", "basket", "fruit", "score", "board"}

    @staticmethod
    def _en_tokens(text: str) -> List[Tuple[str, str, bool]]:
        """(비교용 낱말, 원래 낱말, 문장 첫 낱말인지). 관사·전치사는 빼고 복수형 s는 떼고 비교한다."""
        out, start = [], True
        for m in re.finditer(r"[A-Za-z0-9]+|[.!?]", str(text)):
            w = m.group(0)
            if w in ".!?":
                start = True
                continue
            lw = w.lower()
            if lw not in KBPrep._EN_STOP:
                out.append((lw[:-1] if len(lw) > 3 and lw.endswith("s") else lw, w, start))
            start = False
        return out

    @staticmethod
    def _en_words(text: str) -> List[str]:
        return [w for w, _, _ in KBPrep._en_tokens(text)]

    def _load_aliases(self) -> None:
        from pathlib import Path
        self.alias_index: List[Tuple[Tuple[str, ...], str, str]] = []
        path = Path(os.environ.get("KB_ALIASES", Path(__file__).resolve().parents[1] / "data" / "kb_aliases.json"))
        try:
            data = json.loads(path.read_text(encoding="utf-8")).get("aliases", {}) if path.exists() else {}
            for ko, v in data.items():
                # 이미 영문인 이름·섹션 ID·파일 이름은 별칭이 필요 없다(검색 질의에 그대로 있음)
                if not isinstance(ko, str) or not re.search(r"[가-힣]", ko) or re.search(r"(?:PY|PR|FIN)-\d{2}|\.(?:csv|md|py)\b", ko):
                    continue
                for en in (v.get("en", []) if isinstance(v, dict) else []):
                    words = tuple(self._en_words(en)) if isinstance(en, str) else ()
                    if not words or re.search(r"(?:py|pr|fin)\d{2}", "".join(words)):
                        continue
                    if len(words) == 1 and (words[0] in self._EN_GENERIC or len(words[0]) < 4):
                        continue
                    self.alias_index.append((words, en, ko))
        except Exception as e:  # 사전이 깨져도 서버 시작은 막지 않는다(C-8만 꺼짐)
            log.warning("kb_aliases_load_failed type=%s", type(e).__name__)
            self.alias_index = []
        self.alias_index.sort(key=lambda x: -len(x[0]))
        log.info("kb_aliases_loaded names=%d phrases=%d", len({k for _, _, k in self.alias_index}), len(self.alias_index))
        self._load_offline()

    # ---- v31 오프라인 자료: 섹션별 '다루는 것/다루지 않는 것'(COV), 사실 표(FACT) ----
    def _load_offline(self) -> None:
        from pathlib import Path
        base = Path(__file__).resolve().parents[1] / "data"
        self.coverage_items: Dict[str, Dict[str, list]] = {}
        self.fact_rows: Dict[str, list] = {}
        try:
            path = Path(os.environ.get("KB_COVERAGE", base / "kb_coverage.json"))
            cov = json.loads(path.read_text(encoding="utf-8")).get("coverage", {}) if path.exists() else {}
            for sid, v in cov.items():
                self.coverage_items[sid] = {
                    k: [(_flat_text(it["s"]), _bigrams(it["q"])) for it in v.get(k, []) if isinstance(it, dict) and it.get("s") and it.get("q")]
                    for k in ("covers", "not_covered")}
            path = Path(os.environ.get("KB_FACTS", base / "kb_facts.json"))
            facts = json.loads(path.read_text(encoding="utf-8")).get("facts", {}) if path.exists() else {}
            for sid, rows in facts.items():
                out = []
                for f in rows:
                    vals = _class_values(f.get("v", ""))
                    if vals and f.get("s") and f.get("p"):
                        out.append((_flat_text(f["s"]), _bigrams(f["p"]), vals))
                self.fact_rows[sid] = out
        except Exception as e:  # 자료가 깨져도 서버 시작은 막지 않는다(이 기능만 꺼짐)
            log.warning("kb_offline_load_failed type=%s", type(e).__name__)
            self.coverage_items, self.fact_rows = {}, {}
        log.info("kb_offline_loaded coverage_sections=%d fact_rows=%d", len(self.coverage_items),
                 sum(len(v) for v in self.fact_rows.values()))

    def cover_verdict(self, asked: str, sids: List[str], question: str = "") -> str:
        """묻는 항목이 [자료] 섹션들의 '다루는 것'과 '다루지 않는 것' 중 어디에 맞는가: covered | not_covered | unknown.
        항목의 대상 이름이 묻는 항목(또는 질문)에 글자 그대로 있어야 하고, 물을 거리의 글자 2-gram이 COV_MIN 이상 들어 있어야 맞는다고 본다."""
        if not (_on("COV") and self.coverage_items):
            return "unknown"
        a_flat = _flat_text(asked)
        ctx_flat = a_flat + _flat_text(question)
        a_bi = _bigrams(asked)
        best = {"covers": 0.0, "not_covered": 0.0}
        for sid in sids:
            for kind in ("covers", "not_covered"):
                for s_flat, q_bi in self.coverage_items.get(sid, {}).get(kind, []):
                    if not s_flat or s_flat not in ctx_flat or not q_bi:
                        continue
                    best[kind] = max(best[kind], len(q_bi & a_bi) / len(q_bi))
        if best["not_covered"] >= COV_MIN and best["not_covered"] > best["covers"]:
            return "not_covered"
        return "covered" if best["covers"] >= COV_MIN else "unknown"

    def fact_conflicts(self, candidate: str, sids: List[str], question: str = "") -> List[str]:
        """후보 답변에서 사실 표와 어긋나는 값 구절. 같은 대상(절에 이름이 없으면 질문의 대상)의 같은 속성(글자 2-gram이 하나 이상 겹침)을
        말하면서, 같은 종류의 값(금액·비율·나이·시각·개월 등)이 사실 표의 어느 값과도 맞지 않을 때만. 질문·대화에 나온 숫자는 뺀다.
        E-11(자료 어디에도 없는 값)과 달리, 자료에 있는 값이 다른 대상·속성에 잘못 붙은 경우를 잡는다."""
        if not (_on("FACT") and self.fact_rows):
            return []
        rows = [r for sid in sids for r in self.fact_rows.get(sid, [])]
        if not rows:
            return []
        q_flat = _flat_text(question)
        q_nums = set(re.findall(r"\d+(?:\.\d+)?", _SID_REF.sub(" ", str(question)).replace(",", "")))
        q_vals = _class_values(question)  # 질문·대화에 나온 값은 같은 정규화로 비교해 뺀다('6만 원' = won60000)
        out = []
        cand = _SID_REF.sub(" ", candidate)
        for clause in re.split(r"(?<=[.!?])\s+|,\s+|\s+(?:그리고|따라서|그래서)\s+", cand):
            c_flat, c_bi = _flat_text(clause), _bigrams(clause)
            named = [r for r in rows if r[0] and r[0] in c_flat]
            rel = named or [r for r in rows if r[0] and r[0] in q_flat]
            rel = [r for r in rel if r[1] & c_bi]
            if not rel:
                continue
            for start, end, cls, val in _clause_values(clause):
                same = [r for r in rel if cls in r[2]]
                if not same:
                    continue
                allowed = set().union(*[r[2][cls] for r in same])
                raw = re.sub(r"[^\d.]", "", val.replace("p", "."))
                if val in allowed or raw in q_nums or val in q_vals.get(cls, set()):
                    continue
                out.append(_value_phrase(clause, start, end))
        return list(dict.fromkeys(x for x in out if x))

    def alias_matches(self, text: str) -> List[Tuple[str, str]]:
        """영어 글에서 찾은 (영어 표현, 한국어 이름). 긴 표현부터 찾고, 이미 쓴 낱말은 짧은 표현이 다시 쓰지 않는다.
        한 낱말짜리 표현은 문장 중간에 대문자로 시작하는 이름일 때만 쓴다('finish' 같은 일반 낱말 오탐 방지)."""
        if not (_on("C8") and getattr(self, "alias_index", None)):
            return []
        toks = self._en_tokens(text)
        words = [w for w, _, _ in toks]
        used, out, seen = set(), [], set()
        for ws, en, ko in self.alias_index:
            if ko in seen:
                continue
            n = len(ws)
            for i in range(len(words) - n + 1):
                if tuple(words[i:i + n]) != ws or used & set(range(i, i + n)):
                    continue
                if n == 1 and (not toks[i][1][:1].isupper() or toks[i][2]):
                    continue
                out.append((en, ko))
                seen.add(ko)
                used |= set(range(i, i + n))
                break
        return out

    # ---- 만들기 도우미 ----
    def _dir(self, sid: str):
        from pathlib import Path
        root = Path(self.kb_dir) if self.kb_dir else Path(os.environ.get("KB_DIR", Path(__file__).resolve().parents[1] / "kb"))
        return root / self.sections[sid].subject

    def _md(self, sid: str) -> str:
        return (self._dir(sid) / f"{sid}.md").read_text(encoding="utf-8")

    def _section_tables(self, sid: str) -> Dict[str, str]:
        return {t.name: t.read_text(encoding="utf-8").strip() for t in sorted(self._dir(sid).glob(f"{sid}-table*.csv"))}

    def _strip_disclaimer(self, sid: str, text: str) -> str:
        if not _on("B6") or sid == self.disclaimer_home or not _DISCLAIMER.search(text):
            return text
        return " ".join(s for s in _SENT_SPLIT.split(text) if not _DISCLAIMER.search(s))

    def _units(self, sid: str, hi: int, body: str, tables: Dict[str, str]) -> List[Unit]:
        units, k = [], 0

        def add(kind, text):
            nonlocal k
            text = " ".join(text.split())
            if text:
                k += 1
                units.append(Unit(f"{sid}#{hi}.{k}", sid, hi, kind, text))
        pos = 0
        for m in re.finditer(r"```[a-zA-Z]*\n.*?```", body, re.S):
            self._prose_units(body[pos:m.start()], tables, add)
            add("code", m.group(0))
            pos = m.end()
        self._prose_units(body[pos:], tables, add)
        return units

    @staticmethod
    def _prose_units(text: str, tables: Dict[str, str], add):
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") and not line.startswith("### Q"):
                continue
            mt = re.match(r"\(표:\s*([^)]+?\.csv)\)", line)
            if mt:
                for s in table_sentences(tables.get(mt.group(1).strip(), "")):
                    add("table", s)
                continue
            if line.startswith("### Q"):
                add("q", re.sub(r"^###\s*Q\.?\s*", "", line))
            elif re.match(r"^A\.\s", line):
                add("a", line[2:].strip())
            else:
                for s in _SENT_SPLIT.split(line):
                    add("sent", s)

    def _toks(self, text: str) -> List[str]:
        return tokenize(text) + (num_tokens(text) if _on("C7") else [])

    # ---- 질의 ----
    def query_text(self, query: str) -> str:
        """C-7·C-9: 질의 뒤에 정규화 토큰과 붙인 낱말을 덧붙인다(원문 낱말은 그대로)."""
        extra = []
        if _on("C7"):
            extra += num_tokens(query)
        if _on("C9"):
            toks = query.split()
            for a, b in zip(toks, toks[1:]):
                if len(a) == 1 and re.match(r"[가-힣]", a) and re.match(r"[가-힣]", b):
                    joined = a + re.match(r"[가-힣]+", b).group(0)
                    if joined[:2] in self._kb_text:
                        extra.append(joined)
        return query + ("\n" + " ".join(extra) if extra else "")

    def _rank_docs(self, scores: np.ndarray, sids: List[str], k: int):
        best: Dict[str, float] = {}
        for i in np.argsort(-scores):
            if scores[i] <= 0 or len(best) >= k:
                break
            best.setdefault(sids[i], float(scores[i]))
        return sorted(best.items(), key=lambda x: -x[1])

    def search_faq(self, query: str, k: int = 20):
        if not (_on("A2") and self.faq_bm25):
            return []
        return self._rank_docs(self.faq_bm25.scores(self._toks(self.query_text(query))), [s for s, _ in self.faq], k)

    def search_code(self, query: str, k: int = 20):
        if not _on("F13"):
            return []
        terms = [t.lower() for t in code_terms(query) if t in self.code_vocab]
        if len(set(terms)) < 2:
            return []
        sc = self.code_bm25.scores(terms)
        return [(self.section_ids[i], float(sc[i])) for i in np.argsort(-sc)[:k] if sc[i] > 0]

    def search_anchored(self, query: str, k: int = 20):
        """B-4: 질문에 여러 섹션 공통 낱말 묶음이 있으면, 그 묶음이 든 청크끼리만 청크 BM25로 비교한다."""
        if not (_on("B4") and self.chunk_bm25 is not None):
            return []
        words = re.findall(r"[가-힣a-z0-9]+", normalize(query))
        hits = [g for n in (3, 2) for i in range(len(words) - n + 1) for g in [" ".join(words[i:i + n])] if g in self.name_chunks]
        if not hits:
            return []
        allowed = set().union(*[self.name_chunks[g] for g in hits[:3]])
        sc = self.chunk_bm25.scores(self._toks(self.query_text(query)))
        mask = np.full(len(sc), -1.0)
        idx = np.fromiter(allowed, dtype=int)
        mask[idx] = sc[idx]
        return self._rank_docs(mask, self.chunk_sid, k)

    # ---- A-2 아주 비슷한 FAQ 질문 ----
    def faq_strong(self, last: str, candidates: List[str], th: float = 0.35) -> Optional[str]:
        """마지막 발화가 어떤 FAQ 질문과 글자 3-gram Jaccard th 이상이면 그 섹션(검색 상위 후보 안일 때만)."""
        if not (_on("A2") and self.faq):
            return None
        def g3(x):
            x = re.sub(r"[\s\W_]+", "", normalize(x))
            return {x[i:i + 3] for i in range(len(x) - 2)}
        lg = g3(last)
        if not lg:
            return None
        if not hasattr(self, "_faq_g3"):
            self._faq_g3 = [(sid, g3(t.split(" ", 1)[1] if " " in t else t)) for sid, t in self.faq]
        j, sid = max(((len(lg & g) / max(1, len(lg | g)), sid) for sid, g in self._faq_g3), default=(0.0, None))
        return sid if j >= th and sid in candidates else None

    # ---- A-1 소제목 발췌(기본 끔) ----
    # 주의: 발췌는 청크 글 형식("[섹션ID] 제목 > 소제목", 표는 "[표]")이라 1위 섹션 전문과 형식이 다르다. 켤 때는 형식을 맞출 것.
    def section_excerpt(self, sid: str, query: str, ratio: float = 0.5, max_chunks: int = 2) -> Optional[str]:
        """질문과 맞는 소제목 청크만(청크 BM25 최고점의 ratio 이상, 최대 max_chunks개, 원래 순서)."""
        if not (_on("A1") and self.chunk_bm25 is not None):
            return None
        idx = [i for i, s in enumerate(self.chunk_sid) if s == sid]
        if len(idx) <= 1:
            return None
        sc = self.chunk_bm25.scores(self._toks(self.query_text(query)))
        best = max(sc[i] for i in idx)
        if best <= 0:
            return None
        keep = sorted(sorted(idx, key=lambda i: -sc[i])[:max_chunks])
        keep = [i for i in keep if sc[i] >= ratio * best]
        chunks = self.sections[sid].chunks
        first = idx[0]
        return f"# {sid} {self.sections[sid].title} (질문과 맞는 부분만)\n\n" + "\n\n".join(chunks[i - first].text for i in keep)

    # ---- B-5 같은 사실 ----
    def same_fact_sections(self, quote: str, min_share: float = 0.6) -> List[str]:
        q = _grams(quote)
        if len(q) < 8:
            return []
        out = {}
        for sid, _, g in self.sent_grams:
            if g and len(q & g) / min(len(q), len(g)) >= min_share:
                out[sid] = True
        return list(out)

    def residual_rank(self, query: str, quote: str, sids: List[str]) -> List[Tuple[str, float]]:
        """근거 문장 낱말을 뺀 나머지 질문 낱말로 sids를 섹션 BM25 점수 순으로 줄 세운다."""
        qset = set(self._toks(quote))
        rest = [t for t in self._toks(self.query_text(query)) if t not in qset]
        if not rest or self.section_bm25 is None:
            return []
        sc = self.section_bm25.scores(rest)
        pos = {s: i for i, s in enumerate(self.section_ids)}
        return sorted(((s, float(sc[pos[s]])) for s in sids), key=lambda x: -x[1])

    # ---- D-10·F-14(기본 끔) ----
    # 주의: coverage는 전처리 색인 글 기준이라 LLM에게 보인 [자료]와 다를 수 있다(B-6로 뺀 문장, A-3 표 문장 포함).
    def coverage(self, asked: str, sids: List[str]) -> float:
        """묻는 항목(asked)의 글자 2-gram 중 [자료] 섹션들(카드: 본문·표 문장·Q)에 있는 비율."""
        a = re.sub(r"[\s\W_]+", "", normalize(asked))
        grams = {a[i:i + 2] for i in range(len(a) - 1)}
        if not grams:
            return 1.0
        text = re.sub(r"[\s\W_]+", "", " ".join(normalize(self.section_index_text[s]) for s in sids if s in self.section_index_text))
        return sum(g in text for g in grams) / len(grams)

    # ---- E-11 ----
    def wrong_value_spans(self, candidate: str, sids: List[str], question: str = "", require_known: bool = False) -> List[str]:
        """후보 답변에서 [자료] 섹션들과 질문 어디에도 없는 숫자+단위 값을, 앞 수식어 두 낱말까지 붙여 짧은 구간으로 돌려준다."""
        known = set().union(*[self.section_values.get(s, set()) for s in sids] or [set()]) | value_keys(question)
        q_nums = set(re.findall(r"\d+(?:\.\d+)?", _SID_REF.sub(" ", str(question)).replace(",", "")))
        spans = []
        candidate = _SID_REF.sub(" ", candidate)
        cand_keys = value_keys(candidate)
        if require_known and cand_keys and not (cand_keys & known):
            return []  # 검색이 약한 입력(영어)에서 후보 답변의 값이 [자료]에서 하나도 확인되지 않으면 엉뚱한 섹션일 수 있어 다듬지 않는다
        for clause in re.split(r"(?<=[.!?])\s+|,\s+|\s+(?:그리고|따라서|그래서)\s+", candidate):
            for m in _VALUE.finditer(clause):
                keys = value_keys(m.group(0))
                if not keys or keys & known or m.group(1).replace(",", "") in q_nums:
                    continue  # 자료에 있는 값이거나 질문에 나온 숫자(단위가 달라도)는 틀린 값으로 보지 않는다
                if re.match(r"\s*(?:으로|로|을|를)?\s*(?:나누|곱하|곱해|더하|더해|빼면|빼서)", clause[m.end():]):
                    continue  # 계산 과정의 피연산자(예: 100으로 나누면)는 사실 주장이 아니다
                start = m.start()
                tok_s = clause.rfind(" ", 0, start) + 1
                tok_e = clause.find(" ", m.end())
                token = clause[tok_s: tok_e if tok_e >= 0 else len(clause)]
                if re.search(r"[A-Za-z_(]", token):  # 코드 속 값(예: random.seed(42))은 그 코드 토큰 전체
                    token = re.sub(r"[`'\"“”‘’]", "", token)
                    spans.append(_PARTICLE_TAIL.sub("", re.sub(r"[.,!?]+$", "", token)))
                    continue
                left = clause[:start].rstrip().split(" ")
                lead = []
                for w in reversed(left[-2:]):  # 앞 수식어는 두 낱말까지, 앞 절(…므로·…면·…고)에서는 끊는다
                    if not w or re.search(r"(?:[.!?]|므로|면|고|서|며|해)$", w):
                        break
                    lead.insert(0, w)
                end = m.end()
                tail = re.match(r"[%가-힣]*", clause[end:])
                word = m.group(0) + (tail.group(0) if tail else "")
                if word.endswith("씩"):  # 'N씩 + 동사'는 뒤 동사까지가 한 주장
                    nxt = clause[end + len(tail.group(0) if tail else ""):].split()
                    spans.append(" ".join([word] + nxt[:1]).strip())
                    continue
                word = _PARTICLE_TAIL.sub("", word) or word
                if re.search(r"[0-9][천만]?$", word) and re.match(r"\s*원", clause[end + len(tail.group(0) if tail else ""):]):
                    word += " 원"  # '1천 원'처럼 단위 '원'이 띄어 쓰인 금액
                spans.append(" ".join(lead + [word]).strip())
        return list(dict.fromkeys(s for s in spans if s))
