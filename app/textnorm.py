"""검색·분류용 텍스트 정규화와 토큰화.

외부 사전·모델 없이 동작한다(문자 n-gram). 규정상 형태소 분석기는 허용되지만
모델 파일 포함 여부가 애매하므로 쓰지 않는다.
"""
import re
import unicodedata
from typing import Dict, List

# 낱글자 띄어쓰기 우회("시 스 템", "c o o l i n g")를 붙인다: 한 글자 토큰이 3개 이상 이어진 구간.
_SPACED = re.compile(r"(?<!\S)\S(?: \S){2,}(?!\S)")
_WORD = re.compile(r"[0-9a-z_]+|[가-힣]+")
_HANGUL = re.compile(r"[가-힣]")

# LXP 화면의 빠른 질문 버튼 문구(SPEC §2). 검색 질의에서는 정보가 없으므로 뺀다.
QUICK_BUTTON_PHRASES = (
    "학습자료의 내용을 요약해 주세요.",
    "영상의 내용을 요약해주세요.",
    "실습 도중 어려운 부분이 있어요. 도움을 받을 수 있을까요?",
    "이 퀴즈를 푸는 데 힌트를 받을 수 있을까요?",
    "이 코드를 설명해주세요.",
    "문제 의도를 자세히 알고 싶어요.",
    "힌트를 알려주세요.",
    "문제를 푸는 방법을 차근차근 설명해주세요.",
)


def strip_invisible(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", strip_invisible(str(s)))
    s = re.sub(r"[ \t]+", " ", s)
    s = _SPACED.sub(lambda m: m.group(0).replace(" ", ""), s)
    return s.lower()


def _compact(s: str) -> str:
    return re.sub(r"[\s.?!]", "", normalize(s))


_QUICK = {_compact(p) for p in QUICK_BUTTON_PHRASES}


def is_quick_button(s: str) -> bool:
    return _compact(s) in _QUICK


def tokenize(s: str, ngrams=(2, 3)) -> List[str]:
    """영문·숫자는 단어 단위, 한글은 단어 + 문자 2·3-gram."""
    out: List[str] = []
    for w in _WORD.findall(normalize(s)):
        out.append(w)
        if _HANGUL.match(w):
            for n in ngrams:
                out.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return out


def build_query(messages: List[Dict[str, str]], last_weight: int = 3) -> str:
    """대화 전체를 BM25 질의로 만든다. 마지막 학습자 발화에 가중치를 준다.

    공개 세트 실측: 마지막 발화만 쓰면 Recall@1 0.74, 대화 전체를 쓰면 0.88.
    """
    if not messages:
        return ""
    last = messages[-1]["content"]
    rest = [m["content"] for m in messages[:-1] if not is_quick_button(m["content"])]
    return " ".join([last] * last_weight + rest)


def build_dense_query(messages: List[Dict[str, str]], max_chars: int = 2000) -> str:
    """임베딩 질의: 마지막 발화를 앞에 두고 이전 학습자 발화를 최근 순으로 붙인다."""
    if not messages:
        return ""
    users = [m["content"] for m in messages if m["role"] == "user" and not is_quick_button(m["content"])]
    last = messages[-1]["content"]
    prev = users[:-1][::-1]
    return "\n".join([last, *prev])[:max_chars]
