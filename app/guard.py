"""/guard 처리. (골격 단계: 고정 판정. 입력 분류·후보 답변 판정은 다음 단계에서 추가)"""
from typing import Any, Dict, List, Optional

from .schemas import guard_fallback
from .state import AppState


async def handle_guard(state: AppState, messages: List[Dict[str, str]],
                       candidate: Optional[str], deadline: float) -> Dict[str, Any]:
    return guard_fallback(candidate is not None)
