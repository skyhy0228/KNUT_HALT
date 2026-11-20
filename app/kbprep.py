"""v23 KB 전처리(KBPREP). 원본 kb/는 그대로 두고, 서버 시작 때 파생 색인을 만든다(LLM·외부 자료 없음, 결정적).

A-1 계층 자료      : 2·3위 섹션은 질문과 맞는 소제목 청크만 [자료]에 넣는다(1위 섹션은 전문). → tutor._context
A-2 FAQ 색인       : '### Q.' 질문(섹션 제목 포함)을 따로 색인한다. → 인용 우선 섹션 순위에 더함
A-3 표 문장화      : CSV 행을 '열 이름: 값' 문장으로 풀어 검색 색인에 넣는다(자료·인용에는 원본 표 그대로).
B-4 이름 고정 청크 : 여러 섹션에 나오는 낱말 묶음(예: 과일 바구니)이 질문에 있으면 그 낱말이 든 청크끼리만 비교한 순위.
B-5 같은 사실 묶음 : 답의 근거 문장과 거의 같은 문장이 다른 섹션에도 있으면, 그 사실 낱말을 뺀 나머지 질문 낱말로 섹션을 고른다.
B-6 상투 문구      : '가상 은행' 안내 문장은 처음 나오는 섹션에만 남기고 다른 섹션의 검색 색인에서 뺀다.
C-7 숫자 정규화    : 금액·비율·나이·시각을 한 표기로 바꾼 토큰(won25000, pct1p3, age12, h22)을 색인·질의에 덧붙인다.
C-9 띄어쓰기       : 띄어 쓴 한 글자 + 낱말('야 구')을 붙인 꼴이 KB에 있으면 붙인 낱말을 질의에 덧붙인다.
D-10·F-14 섹션 카드: 묻는 항목(asked)이 [자료] 섹션들에 얼마나 담겨 있는지 잰다. → tutor의 '자료 밖' 보조 판단
E-11·E-12 사실 단위: 문장 ID(섹션#소제목.번호)와 숫자+단위 값을 모아, 가드가 후보 답변의 틀린 값 구간을 짧게 잡는 데 쓴다.
F-13 코드 색인     : 코드 블록의 식별자·문자열·실행 결과를 따로 색인한다.
(C-8 별칭·영문명 사전은 LLM으로 만들어야 해 운영진 확인 전이라 넣지 않았다. 영어 질문은 v19 번역 검색이 처리한다.)
"""
import csv
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .kb import Section, _split_h2
from .textnorm import normalize, tokenize

KBPREP = os.environ.get("KBPREP", "1") == "1"


# 하위 요소 기본값: 공개 세트에서 해가 확인된 요소는 끈다(2026-11-20 측정).
#  B4 공통 이름 순위: 인용 우선 순위에 넣으면 1위 적중 107→99(조건부로 써도 고침 1·깨짐 2).
#  B5 같은 사실 인용 바꾸기: 저장 출력 600건 재생에서 바뀐 1건이 인용 F1 -1.0.
#  D10 자료 밖 보조 판단: 재생에서 기준 0.5/0.6/0.7 모두 action 깨짐이 고침보다 많음(깨짐 1/2/3, 고침 0/1/1).
#  A1 2·3위 섹션 소제목만 넣기: 공개 세트 교차 검증에서는 해가 없었지만(action 173→176) 실제 채점에서 튜터가
#     v19~v22 10회 평균 87.18(85.47~88.72)에서 v23 2회 85.06·85.06으로 떨어졌다. 튜터 프롬프트의 93%를 바꾸는 요소라 v24에서 끈다.
_DEFAULTS = {"B4": "0", "B5": "0", "D10": "0", "A1": "0"}


def _on(name: str) -> bool:
    return KBPREP and os.environ.get(f"KBPREP_{name}", _DEFAULTS.get(name, "1")) == "1"


# ---------------- C-7 숫자·단위 정규화 ----------------
_MONEY = re.compile(r"(?:(\d+(?:\.\d+)?)\s*만\s*(?:(\d+)\s*천\s*)?|(\d+)\s*천\s*|(\d{1,3}(?:,\d{3})+|\d+)\s*)(?:원|won\b)", re.I)
_PCT = re.compile(r"(\d+)\s*점\s*(\d+)\s*(?:%p?|퍼센트(?:\s*포인트)?)|(\d+(?:\.\d+)?)\s*(?:%p?|퍼센트(?:\s*포인트)?|percent)", re.I)
_AGE_RANGE = re.compile(r"만\s*(\d{1,2})\s*(?:세)?\s*(?:부터|[~∼\-])\s*(?:만\s*)?(\d{1,2})\s*세")
_AGE = re.compile(r"(?:만\s*)?(\d{1,2})\s*(?:세|살)(?![금트대계])")
_HOUR = re.compile(r"(오전|오후|밤|낮|저녁|새벽|아침)?\s*(\d{1,2})\s*시(?![간작험각키도대켜점])")


def _num(v: str) -> str:
    f = float(v)
    s = str(int(f)) if f == int(f) else ("%g" % f)
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


# ---------------- E-11 숫자+단위 값 ----------------
_VALUE = re.compile(r"(?<![\w.])(-?\d+(?:[.,]\d+)?)\s*(%p?|퍼센트|[가-힣]{1,2})?")
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


class KBPrep:
    def __init__(self, sections: Dict[str, Section]):
        from .retrieval import BM25  # 순환 import 방지
        self.sections = sections
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

        # E-11 섹션별 값 묶음
        self.section_values = {sid: set().union(*[value_keys(u.text) for u in self.units[sid]] or [set()])
                               for sid in self.section_ids}

    # ---- 만들기 도우미 ----
    def _dir(self, sid: str):
        from pathlib import Path
        return Path(os.environ.get("KB_DIR", Path(__file__).resolve().parents[1] / "kb")) / self.sections[sid].subject

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
        terms = [t.lower() for t in code_terms(query) if t in self.code_vocab]
        if not (_on("F13") and len(set(terms)) >= 2):
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

    # ---- A-1 소제목 발췌 ----
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

    # ---- D-10·F-14 ----
    def coverage(self, asked: str, sids: List[str]) -> float:
        """묻는 항목(asked)의 글자 2-gram 중 [자료] 섹션들(카드: 본문·표 문장·Q)에 있는 비율."""
        a = re.sub(r"[\s\W_]+", "", normalize(asked))
        grams = {a[i:i + 2] for i in range(len(a) - 1)}
        if not grams:
            return 1.0
        text = re.sub(r"[\s\W_]+", "", " ".join(normalize(self.section_index_text[s]) for s in sids if s in self.section_index_text))
        return sum(g in text for g in grams) / len(grams)

    # ---- E-11 ----
    def wrong_value_spans(self, candidate: str, sids: List[str], question: str = "") -> List[str]:
        """후보 답변에서 [자료] 섹션들과 질문 어디에도 없는 숫자+단위 값을, 앞 수식어 두 낱말까지 붙여 짧은 구간으로 돌려준다."""
        known = set().union(*[self.section_values.get(s, set()) for s in sids] or [set()]) | value_keys(question)
        q_nums = set(re.findall(r"\d+(?:\.\d+)?", _SID_REF.sub(" ", str(question)).replace(",", "")))
        spans = []
        candidate = _SID_REF.sub(" ", candidate)
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
