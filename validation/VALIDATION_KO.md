# 검증 상태 — 2026-10-24

**GPU 학습·추론·smoke·평가를 실행하지 않았다.** A100 40GB 메모리 적합성·실행 시간·실험 성능은 미측정이다. GPU smoke는 박사님 서버에서 실행하도록 큐에 제공한다.

## 완료한 CPU 검증

- 핵심/입력 테스트28개 통과, supervisor 테스트2개 통과: `check_cpu_full.log`.
- 로컬 Python3.10 / torch2.12 / Transformers5.8 / PEFT0.19.1. 재사용 CPU venv에서 torch는 로컬 user-site 패키지이므로 test shell 실행 시 해당 경로를 PYTHONPATH로 명시했다. 새 `.venv` 설치 성공이나 production torch2.11 검증을 대신하지 않는다. 실제 CUDA 초기화 없음 assertion 포함.
- 훈련5대화/142세션/3,338원래 질문 및 고정 SFT/토크나이저/평가 입력 해시 검증 통과: `assets_branch.json`.
- 충분 근거 식 중복 제거 후 대화별496/497/509/411/500, 합계2,413개. 1단계 누적 풀은 보존.
- Git 추적 파일만 새 임시 경로로 복사하고 실제13개 입력을 ZIP에서 가져온 뒤 다섯 config의 assets preflight 통과: `clean_checkout_test.log`. 코드/README가 입력 ZIP으로 덮어써지지 않음 확인.
- 원래 RAW executor/LOO credit/누적 풀/근거 scorer 소스가 부모 A/B 버전과 byte 단위로 동일함 확인: `phase1_source_identity.json`. 추가 session std는 별도 명시된 새 단계다.
- Python compile 및 실행 shell 정적 검증 수행. 개별 로그는 전체 ZIP에 포함하며 Git에는 이 요약을 포함한다.

검증한 동작:

1. 동일 상태 full-action reward/local 평균, 미래 QG 포함, 샘플1 실행 및 무효 no-op. 최고 후보로 갈아끼우지 않음.
2. ADD공간/REPLACE초과예산, phase2 UPDATE/DELETE 거절, 여러 원문 턴이 묶인 항목 전체 삭제.
3. OR/AND 구조를 보존하는 dedup, 세션 population std/재중심화 없음/zero-std=1 처리.
4. 재현 가능한 독립 무작위 분기, 비분기 출력 학습 제외, phase2 midpoint/end score 생략.
5. 모든 경로의 동시 세션끝95% 게이트, 다음 세션 전환, phase2 latch 유지, 대화 변경 시1단계 재시작.
6. matched 기준선의 stage2 LOO 유지, 선택된 분기가0개인 세션의 no-update, 근거/비근거 ADD 집계와 분석 상태 구분.
7. 작은 실제 Qwen+PEFT에서 manager만 학습, base/reference 고정, 샘플링 logprob와 teacher-forced logprob 일치, stale probability 거절.
8. 실제 optimizer/adapter 및 phase/bank/cursor/RNG checkpoint 재개 후 연속 실행과 동일 파라미터·뱅크·trace. archive 복구 및 손상 거절.
9. 평가 한 경로 greedy와 session-end 전환, 완료 턴 재생성 없이 재개, 평가 종료 후 공식 근거 집계.
10. supervisor 실패 후 할당 owner 유지/재개/큐 append, 대상 process tree만 취소, 관계 없는 프로세스 보존.

일부 통합 테스트는 유효 action을 스크립트로 공급하면서 실제 작은 Qwen의 logprob/backward를 사용하는 fixture다. 학습 배선·상태 재현성 검사이며 실제 정책의 연구 성능이나 on-policy 실험 결과가 아니다. 실제 무작위 생성 확률 일치 검사는 별도 테스트로 수행했다.

## 아직 하지 않은 검증

- torch2.11 production pin으로 깨끗한 서버 환경 설치 및 전체 베이스 모델 shard를 포함한 full preflight.
- 실제 Qwen4B의 GPU smoke/backward/학습·공식 평가. 샘플 길이별 A10040GB 메모리 상한.
- 실제 4경로의95%전환 간격, 분기모드 비중, p별 시간/성능, 초반ADD율 변화.
- 다중 GPU 분산 및 공유 prefix cache. 이 구현은 실험당1GPU, 순차 생성이다.

첨부 설계의 시뮬/오라클 수치는 새 코드의 실험 결과로 사용하지 않는다. 실제 비교는 제공한 matched/original LOO 조건을 동일 엔진에서 실행하여 수행한다.
