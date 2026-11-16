"""운영진 제공 LLM·임베딩 API 호출.

- 모든 호출은 요청 마감(deadline, event loop 시각) 안에서만 한다.
- 동시 호출 수는 세마포어로 제한한다. 대기 시간도 마감에 포함된다.
- 프롬프트·응답 본문은 로그에 남기지 않는다(예외 종류만).
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from .config import Settings

log = logging.getLogger("halt")

MIN_CALL_S = 2.0      # 남은 시간이 이보다 적으면 호출하지 않는다.
RETRY_MIN_S = 8.0     # 재시도는 남은 시간이 이 이상일 때만.
MARGIN_S = 0.5        # 응답 직렬화·전송 여유.
HEDGE_MIN_S = 4.0     # 헤지 요청은 남은 시간이 이 이상일 때만 보낸다.


def remaining(deadline: float) -> float:
    return deadline - asyncio.get_running_loop().time()


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        # OPENAI_BASE_URL·OPENAI_API_KEY는 환경변수에서 읽힌다. 재시도는 직접 관리한다.
        self.chat = AsyncOpenAI(timeout=settings.llm_call_timeout_s, max_retries=0)
        self.embed = AsyncOpenAI(base_url=settings.embedding_base_url or None,
                                 api_key=settings.embedding_api_key or None,
                                 timeout=30, max_retries=2)
        self.sem = asyncio.Semaphore(settings.llm_concurrency)

    async def close(self) -> None:
        await self.chat.close()
        await self.embed.close()

    async def _chat_once(self, messages, timeout: float, max_tokens: int, json_mode: bool) -> str:
        kwargs: Dict[str, Any] = dict(model=self.s.chat_model, messages=messages,
                                      max_completion_tokens=max_tokens,
                                      extra_body={"reasoning_effort": self.s.reasoning_effort})
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["timeout"] = timeout  # 요청 단위 상한(클라이언트 기본 상한보다 우선)

        async def call():
            async with self.sem:
                r = await self.chat.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
        return await asyncio.wait_for(call(), timeout=timeout)

    async def _json_once(self, messages, timeout: float, max_tokens: int) -> Dict[str, Any]:
        obj = json.loads(await self._chat_once(messages, timeout, max_tokens, True))
        if not isinstance(obj, dict):
            raise TypeError("json_not_object")
        return obj

    async def _json_hedged(self, messages, deadline: float, max_tokens: int) -> Dict[str, Any]:
        """헤지 요청: 첫 호출이 LLM_HEDGE_AFTER_S초 안에 끝나지 않으면 같은 요청을 한 번 더 보내고
        먼저 성공한 응답을 쓴다(나머지는 취소). 운영진 API는 같은 크기의 요청도 3~23초로 처리 시간이
        크게 흔들려(2026-11-16 측정) 일부 요청이 시간 초과로 대체 답변이 되는 것을 막는다.
        요청 내용이 같으므로 답변 품질은 바뀌지 않고, 늦어진 요청만 호출이 1회 늘어난다."""
        left = remaining(deadline) - MARGIN_S
        hedge = self.s.llm_hedge_after_s
        if hedge <= 0:
            return await self._json_once(messages, min(left, self.s.llm_call_timeout_s), max_tokens)
        first = asyncio.ensure_future(self._json_once(messages, left, max_tokens))
        try:
            return await asyncio.wait_for(asyncio.shield(first), timeout=min(hedge, left))
        except asyncio.TimeoutError:
            if first.done():  # 대기가 끝나는 순간 첫 호출이 끝났다면 그 결과(또는 첫 호출 자체의 오류)
                return first.result()
        left2 = remaining(deadline) - MARGIN_S
        if left2 < HEDGE_MIN_S:
            return await first
        log.warning("llm_hedged")
        second = asyncio.ensure_future(self._json_once(messages, left2, max_tokens))
        pending = {first, second}
        last_exc: BaseException = asyncio.TimeoutError()
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception() is None:
                        return t.result()
                    last_exc = t.exception()
            raise last_exc
        finally:
            for t in pending:
                t.cancel()

    async def chat_json(self, messages: List[Dict[str, str]], deadline: float,
                        max_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """JSON 객체 응답을 받는다. 실패·시간 부족이면 None."""
        max_tokens = max_tokens or self.s.max_completion_tokens
        for attempt in range(2):
            left = remaining(deadline) - MARGIN_S
            if left < (MIN_CALL_S if attempt == 0 else RETRY_MIN_S):
                return None
            try:
                return await self._json_hedged(messages, deadline, max_tokens)
            except asyncio.TimeoutError:
                log.warning("llm_timeout attempt=%d", attempt)
                return None  # 시간 초과 뒤 재시도는 마감을 넘기기 쉽다.
            except json.JSONDecodeError:
                log.warning("llm_json_decode_error attempt=%d", attempt)
            except TypeError:
                log.warning("llm_json_not_object attempt=%d", attempt)
            except Exception as e:  # 네트워크·API 오류
                log.warning("llm_error attempt=%d type=%s", attempt, type(e).__name__)
        return None

    async def embed_texts(self, texts: List[str], batch: int = 64) -> np.ndarray:
        """서버 시작 시 KB 임베딩용(마감 없음)."""
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            r = await self.embed.embeddings.create(model=self.s.embedding_model, input=texts[i:i + batch])
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        return np.asarray(out, dtype=np.float32)

    async def embed_query(self, text: str, deadline: float) -> Optional[np.ndarray]:
        """요청 처리 중 질의 임베딩. 실패하면 None(BM25만 사용). 결과는 저장하지 않는다."""
        left = min(self.s.query_embed_timeout_s, remaining(deadline) - MIN_CALL_S)
        if not text.strip() or left <= 0.2:
            return None
        try:
            r = await asyncio.wait_for(
                self.embed.embeddings.create(model=self.s.embedding_model, input=[text]), timeout=left)
            return np.asarray(r.data[0].embedding, dtype=np.float32)
        except asyncio.TimeoutError:
            log.warning("embed_timeout")
        except Exception as e:
            log.warning("embed_error type=%s", type(e).__name__)
        return None
