"""환경변수와 실행 설정.

비밀값(API 키, TEAM_TOKEN)은 .env에서만 읽고, 로그나 응답에 절대 내보내지 않는다.
"""
import logging
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# 런박스에 이미 있는 환경변수(주소·모델명)는 덮어쓰지 않는다.
load_dotenv(ROOT / ".env", override=False)

log = logging.getLogger("halt")


def clean_token(raw: str) -> str:
    """공백·보이지 않는 문자(예: U+200B)를 제거한다. 제출 토큰은 ASCII만 허용된다(SPEC §9)."""
    return "".join(ch for ch in raw if not ch.isspace() and unicodedata.category(ch) != "Cf")


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    team_token: str
    chat_model: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    kb_dir: Path
    cache_dir: Path
    # 요청당 전체 예산(초). 채점기 제한은 30초이며 네트워크 여유를 둔다.
    request_budget_s: float
    # LLM 호출 1회의 상한(초).
    llm_call_timeout_s: float
    # 동시에 진행할 LLM 호출 수 상한(채점기 최대 동시 요청 64).
    llm_concurrency: int
    # 헤지 요청 시작 시각(초). 0이면 끔. 첫 호출이 이 시간 안에 끝나지 않으면 같은 요청을 한 번 더 보낸다.
    llm_hedge_after_s: float
    reasoning_effort: str
    max_completion_tokens: int
    # 질의 임베딩 호출 상한(초). 넘으면 BM25만 쓴다.
    query_embed_timeout_s: float
    use_dense: bool
    # 요청 처리 검색 방식: bm25 | dense | hybrid.
    # 공개 세트 R@1: bm25 0.891, dense 0.689, hybrid(RRF) 0.824 → 기본 bm25.
    retrieval_mode: str
    # few-shot 예시 풀(배포 자료 파일만). 쉼표로 구분.
    fewshot_files: tuple
    fewshot_k: int


def load_settings() -> Settings:
    raw_token = os.environ.get("TEAM_TOKEN", "")
    token = clean_token(raw_token)
    if not token:
        raise RuntimeError("TEAM_TOKEN이 설정되지 않았습니다(.env 확인).")
    if token != raw_token:
        log.warning("TEAM_TOKEN에서 공백/보이지 않는 문자를 제거했습니다. .env 값을 고쳐 주세요.")
    if not all(33 <= ord(c) <= 126 for c in token):
        log.warning("TEAM_TOKEN에 ASCII가 아닌 문자가 있습니다. 제출 토큰은 영문·숫자·기호만 됩니다.")

    return Settings(
        team_token=token,
        chat_model=os.environ["OPENAI_MODEL"],
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", ""),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        kb_dir=Path(os.environ.get("KB_DIR", ROOT / "kb")),
        cache_dir=Path(os.environ.get("CACHE_DIR", ROOT / "cache")),
        request_budget_s=_float("REQUEST_BUDGET_S", 25.0),
        llm_call_timeout_s=_float("LLM_CALL_TIMEOUT_S", 20.0),
        llm_concurrency=int(_float("LLM_CONCURRENCY", 64)),
        llm_hedge_after_s=_float("LLM_HEDGE_AFTER_S", 12.0),
        reasoning_effort=os.environ.get("REASONING_EFFORT", "low"),
        max_completion_tokens=int(_float("MAX_COMPLETION_TOKENS", 2000)),
        query_embed_timeout_s=_float("QUERY_EMBED_TIMEOUT_S", 3.0),
        use_dense=os.environ.get("USE_DENSE", "1") != "0",
        retrieval_mode=os.environ.get("RETRIEVAL_MODE", "bm25"),
        fewshot_files=tuple(Path(p.strip()) for p in os.environ.get(
            "FEWSHOT_FILES", f"{ROOT / 'public/tutor.jsonl'},{ROOT / 'public/guard.jsonl'}").split(",") if p.strip()),
        fewshot_k=int(_float("FEWSHOT_K", 6)),
    )
