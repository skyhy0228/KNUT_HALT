# 시연 안내 (2026-11-25 13:00)

- 대상: 시연하는 팀원
- 기준 커밋: `dfc06ea` (v15, `cand/v15`, `main`의 서버 코드, `submission.json`의 commit). 2026-11-18 실제 제출 89.66
- 시연에서 운영진이 확인하는 것(SPEC §8·§9): 마지막 제출 커밋의 **서버 코드**와 **로그를 남기는 코드**
- 팀 점수는 계정별 마지막 제출 중 최고점이다. 시연 커밋은 그 최고점을 낸 계정의 마지막 제출 커밋이다(11-18 기준 이 계정의 `dfc06ea`)
- 마감 전에 다시 제출해 커밋이 바뀌면 이 문서의 커밋과 줄 번호를 다시 확인한다

---

## 1. 시연 전 점검 (당일 오전, 제출 마감 09:00 이후)

```bash
cd /home/elicer/competition
git log -1 --oneline                                   # 지금 체크아웃한 커밋
python3 -c "import json;print(json.load(open('submission.json'))['commit'][:7])"   # 제출 커밋(토큰은 출력하지 않음)
git diff --stat dfc06ea HEAD -- server.py app/ run.sh  # 출력이 없어야 함: 서버 코드가 제출 커밋과 같다
curl -s http://127.0.0.1:8228/health                   # {"ok": true ...}
```

- 문서만 바뀐 커밋이 뒤에 있어도 괜찮다. 위 `git diff`가 비어 있으면 서버 코드는 제출 커밋과 같다.
- 로그를 화면에 보여 주려면 8228 서버를 터미널에서 `./run.sh`로 띄워 둔다(로그가 그 터미널에 나온다).
- 런박스 세션은 최대 2일이라 재시작했을 수 있다. 서버가 꺼져 있으면 `./run.sh`로 다시 띄운다.

## 2. 보여 줄 코드 위치

| 확인 항목 | 위치 (`dfc06ea` 기준) | 설명 |
|---|---|---|
| **접근 로그** | `server.py:52-63` | 경로, 상태 코드, 소요 ms, `content-length`만 기록 |
| **모든 로그 호출** | 아래 명령 | 25개 전부 시간·길이·상태·개수·오류 종류 이름만 기록. 요청·답변 본문, 프롬프트 없음 |
| 인증 | `server.py:66-72` | `secrets.compare_digest`로 비교, 틀리면 401만 |
| 요청 예산 25초 | `server.py:104-145` | `asyncio.wait_for`(116·137행)로 감싸고 넘치면 규칙 대체 응답 |
| 리다이렉트·문서 경로 끔 | `server.py:49` | `redirect_slashes=False`, `/docs` 등 끔 |
| 비밀값 | `app/config.py:16`, `.gitignore:2-3` | `.env`에서만 읽음, `.env`는 커밋 제외 |
| 외부 호출 | `app/llm.py:32-36` | 운영진 주소(환경변수 `OPENAI_BASE_URL`, `EMBEDDING_BASE_URL`)로만 호출 |
| 범주 → 행동 표 | `app/policy.py:288` `action_for`, `:294` `decision_for` | LLM은 범주만, action·decision은 코드 |
| 보정 규칙 | `app/policy.py:143` `rule_adjust` | 공개 세트 라벨 분석으로 만든 일반 언어 패턴 |
| 거절 고정 문구 | `app/policy.py:301` `REFUSAL_TEMPLATES` | 범주 단위 문구(문항별 답 아님) |
| 후보 답변의 거절 판단 | `app/policy.py:137`, `:193` `has_refusal_cue` | 거절 문형('~할 수 없어요', '도와주기 어려워요')만 본다(v15) |
| 튜터 처리 | `app/tutor.py:394` `handle_tutor`, `:375` `_route`, `:347` `ANSWER_PROMPT` | 판정 호출 + 답변 호출 병렬, 합치기 |
| 근거 문장 검증 | `app/tutor.py:129` `_verified_quotes`, `:157` `_with_quote`, `:180` `_seek_scaffold` | 복사한 문장이 원문에 있는지 Python이 확인 |
| 가드 처리 | `app/guard.py:229` `handle_guard`, `:257` `_decide`, `:217` `_split_decision`, `:401` `_vote_input` | 문제 요구 나열, 3표 다수결 |
| safe 결정 | `app/guard.py:73` `ASK_SAFE`, `:282-286` | LLM에 묻지 않고, 막아야 할 입력에 거절 없이 답했으면 unsafe(v15) |
| 근거 없는 구간 | `app/guard.py:349` `_consensus_spans`, `:167` `_trim_topic`(`:164` 최소 길이) | 표 사이 medoid 합의, 앞 주제어 떼기 |
| LLM 실패 시 대체 | `app/tutor.py:271` `rule_only_tutor`, `app/guard.py:191` `rule_only_guard` | 규칙으로 범주를 정해 막을 요청은 막음 |
| 임베딩 생성·캐시 | `app/state.py:27-52`, `app/retrieval.py:113-127` | 서버 시작 시 운영진 API로 생성, SHA-256 키로 캐시 확인 |
| 디스크 쓰기 | `app/retrieval.py:125-126` | 서버가 쓰는 파일은 KB 임베딩 캐시 두 개뿐 |
| few-shot 예시 풀 | `app/examples.py:78` `ExamplePool`, `:125`, `:131` | 배포된 공개 세트 파일만 사용. 125행·131행 설명은 Q&A 참고 |

모든 로그 호출을 한 번에 보여 주는 명령:

```bash
git grep -nE 'log\.(info|warning|error|exception|debug)\(' dfc06ea -- server.py app/
git grep -nE '^\s*print\(' dfc06ea -- server.py app/    # 출력 없음
```

## 3. 시연 순서 제안 (약 5분)

1. 1장의 점검 명령으로 실행 중인 코드가 제출 커밋과 같음을 보여 준다.
2. 공개 세트 질문 하나로 튜터를 호출한다(토큰은 화면에 나오지 않는다).

   ```bash
   python3 - <<'PY'
   import json, urllib.request
   from dotenv import dotenv_values
   tok = "".join(c for c in (dotenv_values(".env").get("TEAM_TOKEN") or "") if c.isprintable() and not c.isspace())
   item = json.loads(open("public/tutor.jsonl", encoding="utf-8").readline())
   req = urllib.request.Request("http://127.0.0.1:8228/tutor", data=json.dumps({"messages": item["messages"]}).encode(),
                                headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
   print(json.dumps(json.loads(urllib.request.urlopen(req, timeout=35).read()), ensure_ascii=False, indent=2))
   PY
   ```

3. 서버 터미널의 마지막 로그 줄이 `req path=/tutor status=200 ms=… in_bytes=…` 형식뿐임을 보여 준다.
4. 2장의 표 순서대로 로그 코드 → 인증·예산 → 정책 표 → 튜터·가드 → 임베딩 절차를 연다.

시연에 쓰는 질문은 공개 세트나 그 자리에서 만든 질문만 쓴다.

## 4. 예상 질문과 답

| 질문 | 답 |
|---|---|
| 공개 세트를 few-shot으로 쓰는 것은 하드코딩이 아닌가? | SPEC §7.1이 허용한다(공개 세트를 예시로 쓰고, 비슷한 예시를 유사도로 찾아 넣는 것). 채점 요청은 풀에 넣지 않는다 |
| `examples.py:125`에서 같은 질문을 건너뛰는 이유는? | 공개 세트로 로컬 평가할 때 채점 문항 자신이 예시로 들어가는 누수를 막는 장치다. 비공개 문항은 공개 세트와 겹치지 않아 채점에는 영향이 없다 |
| `examples.py:131`에서 unsafe 예시를 빼는 이유는? | v15부터 가드는 LLM에 safe를 묻지 않는다. 이때 공격에 응한 후보 예시가 'grounded: true'로만 보이면 근거 판정을 잘못 배우게 되므로 판정 예시에서 뺀다 |
| safe는 LLM이 판단하지 않나? | 공개 세트 후보 답변 86건에서 unsafe는 입력이 막아야 할 요청일 때뿐이었다(19/19). 그래서 safe는 '막아야 할 입력에 거절 없이 답했는가'로 코드가 정하고, LLM은 근거 여부만 판정한다 |
| 정규식 보정·거절 표현 목록은 하드코딩이 아닌가? | 공개 데이터 라벨을 분석해 만든 일반 언어 패턴이다(운영진 확인 항목). 문항별 답이 아니고 채점 요청을 쓰지 않는다 |
| 채점 요청을 저장하거나 재사용하나? | 아니다. 요청 사이 상태·캐시가 없고, 서버가 디스크에 쓰는 것은 KB 임베딩 캐시뿐이다 |
| 채점 요청인지 알아보고 다르게 동작하나? | 아니다. 인증 헤더 외의 헤더나 요청 패턴으로 분기하지 않는다 |
| 가드 3회·튜터 2회 호출과 헤지는 과도한 호출이 아닌가? | 요청 하나에 병렬로 가드 3회, 튜터 2회(답변 + 범주 판정)다. 헤지는 12초가 지나도 응답이 없을 때만 같은 요청을 한 번 더 보낸다. v14 부하 테스트(동시 64, 300건)에서 128회였고, v15 실제 채점은 93초에 시간 초과 0이었다 |
| 임베딩은 어디서 만들었나? | 런박스 안에서 서버를 시작할 때 운영진 API로 만들고 캐시한다. 외부에서 만든 파일은 반입하지 않았다. 요청 처리 중 검색은 BM25만 쓴다 |
| 왜 임베딩이 아니라 BM25인가? | 공개 세트 근거 섹션 R@1: BM25 0.891, 하이브리드 0.824, 임베딩 0.689. 11-18에 임베딩 설정 630가지(차원·단위·질의·결합)를 다시 비교했는데, 한쪽 문항으로 고르고 다른 쪽으로 잰 교차 선택 기준 차이가 1문항이라 바꾸지 않았다 |
| 추론 수준은? | `low`. `medium`은 응답 21~25초, 시간 초과, JSON 잘림이 생겨 철회했다 |
| 로컬 점수와 실제 점수가 왜 다른가? | 비공개 문항과 채점 판정 모델이 다르고, 제출마다 문항 표본이 바뀌어 1~2점 흔들린다 |
