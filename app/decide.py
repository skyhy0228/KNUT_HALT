"""v27 코드 판단(DECIDE=code, 기본): 판단은 코드가 먼저, LLM은 규칙이 정하지 못한 문항의 재료만 낸다.

0단계(코드, LLM 없음) — 키워드 규칙(policy.rule_category)이 범주를 정하면 그대로 확정한다.
    공개 세트 400문항: 규칙이 범주를 정한 139문항 모두 맞음, 정상 질문 212개 오탐 0, 정상이 아닌 188문항 중 74%를 규칙이 정한다
    (v26 규칙은 127문항 중 126개 → v27 규칙 6개 추가). 규칙을 만든 자료의 수치라 비공개 세트에서는 더 낮을 수 있다.
    - 튜터: 고정 문구 범주(탈옥·유해·주제 이탈·정서 위기·뜻 없음·과제 대행)면 LLM을 부르지 않고 고정 문구로 답한다.
    - 가드: 차단 범주면 후보 답변이 있어도 관례(safe=false·근거 있음·구간 없음, 공개 19/19)로 LLM 없이 판정한다.
      허용 범주인데 후보 답변이 없어도 LLM을 부르지 않는다.
1단계(LLM) — 규칙이 정하지 못한 문항만. 튜터는 답변 호출 1회(입력 판정 호출을 부르지 않음), 가드는 3표(근거 판정·구간).
2단계(코드) — LLM이 낸 범주를 그대로 쓰지 않는다.
    - 규칙 범주와 LLM 범주가 다르면 고정 우선순위로 고른다(아래 PRIORITY).
    - '정답만 요구'는 규칙 표현(정답만·답만·설명 없이 …)이 있을 때만 인정한다(공개 세트 LLM 오판: 퀴즈로 만들어 줘, 정답은 뭐야).
    - 자료 밖 판단은 코드로 옮기지 않았다: LLM이 뽑은 '묻는 항목'을 자료 글자와 대조하면 저장 출력 600건에서
      가장 나은 조합도 고침 1·깨짐 6(항상 적용하면 고침 13·깨짐 196) — 묻는 항목에 동사·영어·다른 표현이 섞인다.
DECIDE=llm이면 v26과 같다.
"""
import os
from typing import Optional

from .policy import _ANSWER_ONLY, _ANSWER_ONLY_EN, rule_category
from .textnorm import normalize

DECIDE_CODE = os.environ.get("DECIDE", "code") == "code"

# 규칙이 정하면 LLM 없이 고정 문구로 답하는 튜터 범주(응대 방식이 refuse·redirect·escalate·clarify·대행 scaffold).
TEMPLATE_CATEGORIES = frozenset({"jailbreak_injection", "harmful_content", "off_topic", "emotional_distress",
                                 "low_signal_input", "academic_dishonesty"})
# 범주 우선순위(policy.CATEGORY_GUIDE의 [우선순위]와 같다): 앞일수록 강하다.
PRIORITY = ("emotional_distress", "harmful_content", "jailbreak_injection", "academic_dishonesty", "off_topic",
            "answer_seeking", "out_of_material", "pii_disclosure", "low_signal_input", "normal")


def rule_decision(last_text: str) -> Optional[str]:
    """규칙만으로 정한 범주(정상이면 None). DECIDE=llm이면 늘 None."""
    if not DECIDE_CODE:
        return None
    c = rule_category(last_text)
    return None if c == "normal" else c


def stronger(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """두 범주 중 우선순위가 높은 쪽."""
    if not a:
        return b
    if not b:
        return a
    ia = PRIORITY.index(a) if a in PRIORITY else len(PRIORITY)
    ib = PRIORITY.index(b) if b in PRIORITY else len(PRIORITY)
    return a if ia <= ib else b


def answer_only(last_text: str) -> bool:
    """'정답만 알려 줘'처럼 풀이 없이 정답만 달라는 규칙 표현이 있는가."""
    t = normalize(last_text)
    return bool(_ANSWER_ONLY.search(t) or _ANSWER_ONLY_EN.search(t))
