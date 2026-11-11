"""KNUT_HALT 서버: POST /tutor, POST /guard (SPEC §3~§5).

설계 원칙
- 토큰이 틀린 요청만 401. 그 밖의 모든 경우(잘못된 JSON, 내부 오류, 시간 초과)는 200 + 규격 JSON.
- 요청당 시간 예산(REQUEST_BUDGET_S, 기본 25초) 안에서 반드시 응답한다.
- 색인·임베딩은 서버 시작 시 준비한다(첫 요청 전에 완료).
- 로그에는 경로·상태 코드·소요 시간·바이트 길이만 남긴다. 요청·답변 본문은 저장하지 않는다(SPEC §8).

실행: ./run.sh  (또는 python3 -m uvicorn server:app --host 0.0.0.0 --port 8228 --no-access-log)
제출 서버 포트는 8228이다(run.sh 기본값). 제출 주소의 /proxy/<포트>와 맞춘다.
"""
import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import load_settings
from app.guard import handle_guard, rule_only_guard
from app.schemas import (guard_fallback, parse_candidate, parse_messages, sanitize_guard,
                         sanitize_tutor, tutor_fallback)
from app.state import build_state
from app.tutor import handle_tutor, rule_only_tutor

log = logging.getLogger("halt")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

MAX_BODY_BYTES = 2 * 1024 * 1024
settings = load_settings()
_expected_auth = f"Bearer {settings.team_token}".encode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = await build_state(settings)
    yield
    await app.state.ctx.llm.close()


app = FastAPI(lifespan=lifespan, redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # 문항 내용이 아닌 정보만 기록한다.
        log.info("req path=%s status=%d ms=%d in_bytes=%s", request.url.path, status,
                 (time.perf_counter() - t0) * 1000, request.headers.get("content-length", "-"))


def _authorized(request: Request) -> bool:
    got = request.headers.get("authorization", "").encode("utf-8", "replace")
    return secrets.compare_digest(got, _expected_auth)


def _unauthorized() -> JSONResponse:
    return JSONResponse({"detail": "invalid token"}, status_code=401)


async def _read_payload(request: Request):
    try:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            log.warning("body_too_large bytes=%d", len(body))
            return None
        return json.loads(body)
    except Exception as e:
        log.warning("bad_request_body type=%s", type(e).__name__)
        return None


def _safe_rule_tutor(request: Request, messages):
    """LLM 경로가 실패했을 때 규칙만으로 답한다. 규칙 처리도 실패하면 기존 대체 응답."""
    try:
        return rule_only_tutor(request.app.state.ctx, messages)
    except Exception as e:
        log.warning("tutor_rule_fallback_error type=%s", type(e).__name__)
        return tutor_fallback()


def _safe_rule_guard(messages, candidate):
    try:
        return rule_only_guard(messages, candidate)
    except Exception as e:
        log.warning("guard_rule_fallback_error type=%s", type(e).__name__)
        return guard_fallback(candidate is not None)


def _deadline() -> float:
    return asyncio.get_running_loop().time() + settings.request_budget_s


@app.post("/tutor")
async def tutor(request: Request):
    if not _authorized(request):
        return _unauthorized()
    payload = await _read_payload(request)
    messages = parse_messages(payload)
    deadline = _deadline()
    try:
        result = await asyncio.wait_for(handle_tutor(request.app.state.ctx, messages, deadline),
                                        timeout=settings.request_budget_s)
    except asyncio.TimeoutError:
        log.warning("tutor_budget_exceeded")
        result = _safe_rule_tutor(request, messages)
    except Exception as e:
        log.warning("tutor_error type=%s", type(e).__name__)
        result = _safe_rule_tutor(request, messages)
    return JSONResponse(sanitize_tutor(result))


@app.post("/guard")
async def guard(request: Request):
    if not _authorized(request):
        return _unauthorized()
    payload = await _read_payload(request)
    messages = parse_messages(payload)
    candidate = parse_candidate(payload)
    has_candidate = candidate is not None
    deadline = _deadline()
    try:
        result = await asyncio.wait_for(handle_guard(request.app.state.ctx, messages, candidate, deadline),
                                        timeout=settings.request_budget_s)
    except asyncio.TimeoutError:
        log.warning("guard_budget_exceeded")
        result = _safe_rule_guard(messages, candidate)
    except Exception as e:
        log.warning("guard_error type=%s", type(e).__name__)
        result = _safe_rule_guard(messages, candidate)
    return JSONResponse(sanitize_guard(result, has_candidate))


@app.get("/health")
async def health(request: Request):
    """준비 상태 확인용(인증 없음, 비밀·문항 정보 없음)."""
    ctx = getattr(request.app.state, "ctx", None)
    return {"ok": ctx is not None, "dense": bool(ctx and ctx.retriever.dense_ready)}
