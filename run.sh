#!/usr/bin/env bash
# 서버 실행. 8080은 VSCode가 쓰므로 기본 8000 포트.
# 접근 로그는 server.py 미들웨어가 경로·상태·시간만 남기므로 uvicorn 접근 로그는 끈다.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m uvicorn server:app \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --loop uvloop --no-access-log --timeout-keep-alive 30 \
  --log-level info
