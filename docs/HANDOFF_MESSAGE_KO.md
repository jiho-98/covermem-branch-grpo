# 박사님께 전달할 메시지

박사님, 앞서 전달드린 선택기 LoRA A/B와 별도로, 회의에서 제안된 **95% 이후 같은 상태 분기 GRPO** 코드를 전달드립니다.

- 별도 저장소: https://github.com/jiho-98/covermem-branch-grpo
- 첨부: `CoverMem_branch_GRPO_full.zip`, `SHA256SUMS.txt`
- 설치/실행 안내: 저장소 `docs/SERVER_RUNBOOK_KO.md`
- 구현 결정과 비교 조건: `docs/METHOD_AND_DECISIONS_KO.md`

이번 실험은 별도 선택기 없이 매니저 LoRA 하나를 학습합니다. 대화마다4경로로 기존 LOO를 진행하고, 세션 끝에 네 뱅크가 모두95% 이상이면 다음 세션부터2단계로 전환합니다. 2단계에서는 기본 K=4,p=1/4로 무작위 선택한 상태만 분기하여, 해당 대화 전체 QG의 중복 제거된 충분 근거 집합으로 행동 전후 순이익을 채점합니다. K개 평균을 뺀 local advantage로 학습하고 실제 뱅크는 샘플1로 진행합니다.

사용자 확인에 따라1단계의 기존 항목 구조와5개 액션은 유지하고, 2단계는 ADD/SKIP/REPLACE만 허용합니다. UPDATE로 합쳐진 항목은 교체 시 항목 전체가 빠집니다. 문서의 업데이트 단위 std정규화를 포함했고, 이를 맞춘 LOO 기준선과 원래 LOO 기준선을 별도 config로 제공합니다.

기본 실행은 `branch_p025.json`, 비교는 `vanilla_matched.json`과 `vanilla_original.json`입니다. p0.5와p1은 추가 실험용으로 준비했습니다. A100 40GB ×16 서버에서 **실험당 GPU1장**을 사용하며 여러 조건을 독립 GPU에 배치할 수 있습니다. 16GPU 분산이나 K개 간 prefix cache는 구현하지 않아 실제 시간은 측정 후 비교해야 합니다.

CPU 테스트28개와 supervisor 테스트2개, 실제 데이터/SFT 입력 검증을 통과했습니다. GPU smoke·학습·평가는 아직 실행하지 않았습니다. 안내서에 따라 환경/모델/full preflight를 준비한 뒤 실행하면, 큐가 GPU smoke → 학습 → 평가 → CPU 분석을 자동으로 진행합니다. 근거/비근거 초기ADD율, 95%전환 차이, 생성/학습 출력 및 시간, 평가 REPLACE 근거 손실도 기록합니다.

완료 후 `TRAIN_COMPLETE.json`, `evaluation/EVAL_COMPLETE.json`, `ANALYSIS_COMPLETE.json`, `analysis/comparison.json`, smoke 결과와 큐 로그/환경/commit 정보를 공유 부탁드립니다. 최종 checkpoint 전체와 배치별trace도 보존 부탁드립니다. 비공개 저장소 접근 권한이 없을 때는 전체 ZIP으로 같은 코드를 실행할 수 있습니다.
