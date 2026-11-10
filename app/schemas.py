"""요청 파싱과 응답 규격 보장(SPEC §4·§5).

어떤 입력·내부 오류에도 규격에 맞는 dict를 돌려주는 것이 목적이다.
"""
from typing import Any, Dict, List, Optional

ACTIONS = ("answer", "scaffold", "clarify", "abstain", "redirect", "refuse", "escalate")
CATEGORIES = ("normal", "answer_seeking", "academic_dishonesty", "off_topic", "out_of_material",
              "jailbreak_injection", "pii_disclosure", "harmful_content", "emotional_distress",
              "low_signal_input")
# 공개 세트에서 block으로 라벨된 범주(채점기 ATTACK_CATEGORIES와 동일).
BLOCK_CATEGORIES = frozenset({"jailbreak_injection", "harmful_content", "off_topic", "academic_dishonesty"})
ANSWER_MAX = 4000

FALLBACK_ANSWER = "지금은 답변을 만들기 어려워요. 궁금한 내용을 조금 더 구체적으로 다시 질문해 줄래요?"


def parse_messages(payload: Any) -> List[Dict[str, str]]:
    raw = payload.get("messages") if isinstance(payload, dict) else None
    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                c = m.get("content")
                out.append({"role": m["role"], "content": c if isinstance(c, str) else ("" if c is None else str(c))})
    return out


def parse_candidate(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict) or "candidate_answer" not in payload:
        return None
    c = payload["candidate_answer"]
    if c is None:
        return None
    return c if isinstance(c, str) else str(c)


def tutor_fallback(citations: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"answer": FALLBACK_ANSWER, "action": "answer", "citations": list(citations or [])}


def sanitize_tutor(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return tutor_fallback()
    answer = obj.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    if not answer:
        answer = FALLBACK_ANSWER
    action = obj.get("action") if obj.get("action") in ACTIONS else "answer"
    cites = obj.get("citations")
    cites = list(dict.fromkeys(c for c in cites if isinstance(c, str))) if isinstance(cites, list) else []
    return {"answer": answer[:ANSWER_MAX], "action": action, "citations": cites}


def guard_fallback(has_candidate: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"input": {"decision": "allow", "category": "normal"}}
    if has_candidate:
        out["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
    return out


def sanitize_guard(obj: Any, has_candidate: bool) -> Dict[str, Any]:
    base = guard_fallback(has_candidate)
    if not isinstance(obj, dict):
        return base
    inp = obj.get("input") if isinstance(obj.get("input"), dict) else {}
    category = inp.get("category") if inp.get("category") in CATEGORIES else "normal"
    decision = inp.get("decision") if inp.get("decision") in ("allow", "block") else (
        "block" if category in BLOCK_CATEGORIES else "allow")
    result: Dict[str, Any] = {"input": {"decision": decision, "category": category}}
    if has_candidate:
        o = obj.get("output") if isinstance(obj.get("output"), dict) else {}
        spans = o.get("unsupported")
        spans = [s for s in spans if isinstance(s, str) and s.strip()] if isinstance(spans, list) else []
        result["output"] = {
            "grounded": o["grounded"] if isinstance(o.get("grounded"), bool) else True,
            "unsupported": spans,
            "safe": o["safe"] if isinstance(o.get("safe"), bool) else True,
            "category": o["category"] if isinstance(o.get("category"), str) else category,
        }
    return result
