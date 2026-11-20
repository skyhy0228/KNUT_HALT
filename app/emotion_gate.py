"""정서 표현 구절 관문(TUTOR_EMO_GATE): '걱정돼' 같은 감정 낱말이 앞뒤 낱말과 함께 어떤 뜻인지 본다.

문제(11-20 측정): 입력 판정 LLM이 '실습 어렵고 걱정됨', '자꾸 막혀서 걱정되는데'처럼 과제에 대한 가벼운 걱정을
emotional_distress로 잡는다(정상 문항 오탐, 튜터는 escalate 고정 문구가 되어 문항 점수 약 0.75 손실).
범주 정의를 고치거나 구분 예시를 넣으면 정서 위기 쪽으로 경계 전체가 움직일 뿐 오탐은 줄지 않았다(26문항×5회: 오탐 8 → 13).

방법: 마지막 학습자 발화에서 감정 낱말과 앞뒤 GATE_WORDS어절만 잘라(구절) 제공 임베딩 모델로 임베딩하고,
예시 풀(배포된 공개 세트 라벨)에서 만든 두 중심점 — 정서 위기 문항의 구절 / 감정 낱말이 있지만 정서 위기가 아닌 문항의 구절 —
과의 코사인 유사도 차이를 점수로 쓴다(>0이면 정서 위기 쪽). 문장 전체를 임베딩하면 수업 질문 부분이 의미를 덮어
구분이 안 된다(그룹 분할 평가: 구절 26/27, 문장 전체 18/27).

- 감정 낱말 목록은 구절의 위치를 찾는 데만 쓴다. 판단은 학습한 중심점이 한다.
- 중심점은 서버 시작 때 예시 풀로 한 번 만든다(임베딩 호출 1회). 채점 요청은 학습에 쓰지 않고 저장하지 않는다.
- 임베딩이 실패하면 관문 없이 동작한다(None).
"""
import logging
import os
import re
from typing import List, Optional

import numpy as np

log = logging.getLogger("halt")

EMO_WORDS = re.compile(r"걱정|불안|어려|막혀|막히|막막|힘들|힘드|힘듦|지쳐|지쳤|우울|버티|견디|싫|희망|무기력|외로|슬퍼|괴로")
GATE_WORDS = int(os.environ.get("EMO_GATE_WORDS", "3"))
MIN_PER_CLASS = 3


def emotion_span(text: str, w: int = GATE_WORDS) -> str:
    """감정 낱말이 든 어절과 그 앞뒤 w어절(겹치면 합침). 감정 낱말이 없으면 ''."""
    words = (text or "").split()
    keep = set()
    for i, wd in enumerate(words):
        if EMO_WORDS.search(wd):
            keep.update(range(max(0, i - w), min(len(words), i + w + 1)))
    return " ".join(words[i] for i in sorted(keep))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _loo_scores(v: np.ndarray, n_pos: int) -> np.ndarray:
    """각 구절을 빼고 만든 중심점으로 그 구절을 매긴 점수(leave-one-out)."""
    pos, neg = v[:n_pos], v[n_pos:]
    s1, s0 = pos.sum(0), neg.sum(0)
    out = []
    for i, x in enumerate(v):
        c1 = _unit(s1 - x if i < n_pos else s1)
        c0 = _unit(s0 - x if i >= n_pos else s0)
        out.append(float(x @ c1 - x @ c0))
    return np.asarray(out)


class EmotionGate:
    """score>0이면 정서 위기 쪽. 올리기는 더 확실할 때만(up_threshold): 예시 풀의 leave-one-out 점수에서
    정서 위기 구절의 최솟값과 그 밖 구절의 최댓값의 중간(둘이 겹치면 0). 내리기 문턱은 0으로 둔다 —
    진짜 정서 위기를 정상으로 내리면 위기 학생에게 일반 답변이 나가므로 보수적으로 한다."""

    def __init__(self, c_distress: np.ndarray, c_other: np.ndarray, n_pos: int, n_neg: int,
                 up_threshold: float = 0.0):
        self.c1 = _unit(c_distress)
        self.c0 = _unit(c_other)
        self.n_pos, self.n_neg = n_pos, n_neg
        self.up_threshold = up_threshold

    @classmethod
    async def build(cls, llm, examples) -> Optional["EmotionGate"]:
        pos: List[str] = []
        neg: List[str] = []
        for e in examples.items:
            sp = emotion_span(e.last)
            if not sp:
                continue
            (pos if e.category == "emotional_distress" else neg).append(sp)
        pos, neg = list(dict.fromkeys(pos)), list(dict.fromkeys(neg))
        if len(pos) < MIN_PER_CLASS or len(neg) < MIN_PER_CLASS:
            log.warning("emo_gate_skipped pos=%d neg=%d", len(pos), len(neg))
            return None
        try:
            v = _unit(await llm.embed_texts(pos + neg))
        except Exception as e:
            log.warning("emo_gate_embed_failed type=%s", type(e).__name__)
            return None
        loo = _loo_scores(v, len(pos))
        lo_pos, hi_neg = float(loo[:len(pos)].min()), float(loo[len(pos):].max())
        up = max(0.0, (lo_pos + hi_neg) / 2) if hi_neg < lo_pos else 0.0
        gate = cls(v[:len(pos)].mean(0), v[len(pos):].mean(0), len(pos), len(neg), up)
        log.info("emo_gate_ready pos=%d neg=%d loo_pos_min=%.3f loo_neg_max=%.3f up=%.3f",
                 len(pos), len(neg), lo_pos, hi_neg, up)
        return gate

    def score_vec(self, v: np.ndarray) -> float:
        v = _unit(np.asarray(v, dtype=np.float32))
        return float(v @ self.c1 - v @ self.c0)

    async def score(self, llm, text: str, deadline: float) -> Optional[float]:
        """>0이면 정서 위기 쪽, <=0이면 가벼운 감정 쪽. 감정 낱말이 없거나 임베딩 실패면 None."""
        sp = emotion_span(text)
        if not sp:
            return None
        v = await llm.embed_query(sp, deadline)
        return None if v is None else self.score_vec(v)
