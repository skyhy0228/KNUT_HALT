#!/usr/bin/env bash
# 서버 실행. 팀 제출 서버 포트는 8228(2026-11-11 8000에서 변경). 8080은 VSCode가 쓰므로 쓰지 않는다.
# 제출 주소: https://<런박스 주소>.tunnel.elice.io/proxy/8228/tutor, …/guard
# 접근 로그는 server.py 미들웨어가 경로·상태·시간만 남기므로 uvicorn 접근 로그는 끈다.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m uvicorn server:app \
  --host 0.0.0.0 --port "${PORT:-8228}" \
  --loop uvloop --no-access-log --timeout-keep-alive 30 \
  --log-level info
