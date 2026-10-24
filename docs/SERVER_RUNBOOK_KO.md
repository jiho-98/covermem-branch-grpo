# A100 40GB ×16 — 분기 GRPO 설치·실행 안내

AI 에이전트는 먼저 [통합 인계서](AI_AGENT_HANDOFF_KO.md)를 읽는다. 자료는 [Google Drive 다운로드·분할 복원 안내](DOWNLOAD_AND_RESTORE_KO.md)에 따라 준비한다. 최신 코드는 Git checkout을 사용하고 복원 ZIP에서는 입력만 가져오는 방법을 권장한다.

새 실험의 기본 조건은 **K=4, p=0.25, 대화별4경로 동시95% 세션끝 게이트**다. A/B 선택기 K=10 코드와 디렉토리를 섞지 않는다. 모든 명령은 이 저장소/패키지 루트에서 실행한다.

## 1. 서버 준비

Linux x86_64, Python3.10을 사용한다. 서버에16장이 있어도 각 실행은 GPU1장/프로세스1개다. 기본 분기/맞춘 기준선/원래 기준선을 GPU3장에 독립 배치할 수 있다. multi-GPU DDP/FSDP/텐서 병렬을 실행하는 패키지는 아니다.

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .
free -h
command -v sbatch || true
```

사용할 GPU의 권한/점유 상태를 서버 운영 규칙에 따라 확인한다. 입력 파일 약0.2GB, 별도 베이스 모델, Python/CUDA 환경, checkpoint와 모든 배치 trace를 저장할 공간이 필요하다. 실제 전체 실행의 RAM/디스크/시간 요구량은 아직 측정하지 않았다. GPU40GB에 들어가는지 첫 smoke에서 검사하며, smoke 성공도 모든 생성 길이의 메모리 상한을 보장하지 않는다.

## 2. 코드와 ZIP 입력

GitHub 저장소는 비공개다. 소유자가 박사님 GitHub 계정에 초대하고 초대를 수락한 뒤 평소 HTTPS/SSH 인증으로 clone한다. Git에는 데이터/어댑터가 없으므로 전체 ZIP도 준비한다.

```bash
git clone https://github.com/jiho-98/covermem-branch-grpo.git
cd covermem-branch-grpo

# 이미 python3.10이 있으면 conda 단계는 생략한다.
conda env create -f environment.yml
conda activate covermem-branch

# 받은 파일의 실제 위치로 변경한다.
python3.10 scripts/import_assets.py --archive /path/to/CoverMem_branch_GRPO_full.zip
python3.10 scripts/import_assets.py --check-only
```

`inputs_verified`, `files: 13`, `gpu_used: false`를 확인한다. 고정 SHA256에 맞는 데이터·토크나이저·SFT 어댑터만 가져오며 ZIP의 코드가 Git checkout을 덮어쓰지 않는다. 기존 입력이 다르면 덮어쓰지 않고 실패한다. 가져오기 중 입력 크기만큼 임시 공간을 추가 사용한다.

GitHub 없이 ZIP만 사용할 수도 있다. ZIP과 `SHA256SUMS.txt`를 같은 폴더에 놓은 뒤:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
name='CoverMem_branch_GRPO_full.zip'
expected=next(line.split()[0] for line in Path('SHA256SUMS.txt').read_text().splitlines() if line.split()[-1]==name)
h=hashlib.sha256()
with Path(name).open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==expected, 'ZIP SHA256 mismatch'
print('ZIP SHA256 OK')
PY
unzip CoverMem_branch_GRPO_full.zip
cd covermem_branch
conda env create -f environment.yml
conda activate covermem-branch
python3.10 scripts/import_assets.py --check-only
```

이전 실행 폴더에 덮어 풀지 않는다. code-only ZIP은 검토용이며 데이터/어댑터가 없어 단독 실행할 수 없다. 이전 `CoverMem_selector_GRPO_AB_full.zip`과 이번 `CoverMem_branch_GRPO_full.zip`을 구분한다.

## 3. 환경 설치와 모델

```bash
bash scripts/setup_env.sh
bash scripts/download_model.sh
```

설치는 루트의 `.venv` 안에 한다. 설치 Python을 명시하려면 `COVERMEM_PYTHON=/path/to/python3.10 bash scripts/setup_env.sh`를 사용한다. 핵심 버전은 아래와 같다.

| 항목 | 값 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.11.0 |
| Transformers | 5.8.0 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| 전체 고정 목록 | `requirements.txt`, `environment.lock.json` |
| 모델 | `Qwen/Qwen3-4B-Instruct-2507` |
| 고정 revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 저장 위치 | `models/Qwen3-4B-Instruct-2507` |

기본 torch CUDA wheel index는 `https://download.pytorch.org/whl/cu130`이다. 서버 드라이버가 이 빌드를 지원하는지 확인한다. `nvcc` 버전만으로 판단하지 않는다. 관리자가 호환 가능한 다른 공식 torch2.11.0 index를 제공하면 설치 전 `COVERMEM_TORCH_INDEX_URL`로 지정한다. 드라이버를 자동 설치하지 않으며 runtime/CUDA 변경 후 기존 checkpoint를 재개하지 않는다.

vLLM/verl 설치는 필요하지 않다. K개 생성은 HF의 순차 실행이며 공유 prefix cache가 없다. 기존 A/B와 같은 모델 파일을 read-only로 공유하려면 각 config의 `paths.base_model`을 실제 경로로 지정할 수 있다. 상대 경로는 패키지 루트 기준이다. 오프라인 서버는 위 revision의 모델 shard/tokenizer/config 파일을 외부에서 받아 지정 경로에 복사한다. SFT 어댑터만으로 실행할 수 없다.

## 4. GPU 없이 검사

```bash
bash scripts/check_cpu.sh configs/branch_p025.json

bash scripts/preflight.sh configs/branch_p025.json
cp validation/preflight.json validation/preflight_branch_p025.json
bash scripts/preflight.sh configs/vanilla_matched.json
cp validation/preflight.json validation/preflight_vanilla_matched.json
bash scripts/preflight.sh configs/vanilla_original.json
cp validation/preflight.json validation/preflight_vanilla_original.json
```

확인할 사항:

- `tests`: 28개 `OK`, supervisor: 2개 `OK`.
- 입력 preflight: 142세션/3,338원래 질문, `assets_validated`, GPU사용없음.
- full preflight: `prepared`, fingerprint 존재, 모델 shard·버전·입력 해시 오류 없음.
- `validation/installed.freeze.txt`, 각 preflight JSON 보존.

중복 제거한2단계 질문은 5대화 각각496/497/509/411/500개다. 1단계 누적 질문을 이 수로 바꾸지 않는다. 사전 검증은 로컬 torch2.12 CPU 환경에서 수행했으며, 박사님 서버의 새 torch2.11 설치 환경에서 위 명령을 다시 통과해야 한다. 모델 해시 검사 중 출력이 잠시 없어도 GPU를 사용 중인 것은 아니다.

## 5. 일반 전용 서버 실행

Slurm 서버는 다음 절을 따른다. 권한을 확보한 GPU 번호로0/1/2를 바꾸고 별도 터미널 또는 tmux 창에서 실행한다. SSH가 끊겨도 실행 세션을 유지하려면 예를 들어 `tmux new -s cm-branch` 안에서 시작하고 Ctrl-b 다음 d로 빠져나온다.

```bash
# GPU0: 기본 실험
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh configs/branch_p025.json orchestration/branch_p025

# GPU1: 액션 제한/정규화를 맞춘 LOO 기준선
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh configs/vanilla_matched.json orchestration/vanilla_matched

# GPU2: 원래 LOO 방식의 같은 엔진 기준선
CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh configs/vanilla_original.json orchestration/vanilla_original
```

각 명령의 자동 순서는 **GPU smoke → 학습 → 완료 checkpoint 평가 → CPU 분석 → 종료**다. 후속 평가를 별도로 시작하거나 partial checkpoint를 평가하지 않는다. 실험별 결과는 `runs/branch_v1/branch_p025`, `vanilla_matched`, `vanilla_original`에 저장한다.

추가 p조건이 필요할 때만 별도 GPU/큐에 실행한다:

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/run.sh configs/branch_p050.json orchestration/branch_p050
CUDA_VISIBLE_DEVICES=4 bash scripts/run.sh configs/branch_p100.json orchestration/branch_p100
```

이 명령들을 안내 작성 환경에서 실제 실행하지 않았다. 16 GPU 전체를 예약하거나 자동으로 모든 config를 실행하는 스크립트는 없다. 한 실험에 GPU가 정확히 한 장만 보이게 해야 하며 `torchrun`을 사용하지 않는다.

## 6. Slurm 서버 실행

먼저 `squeue -u "$USER"`로 기존 할당을 확인한다. **호환되는 기존 할당을 우선 재사용**하며 `CUDA_VISIBLE_DEVICES`는 Slurm이 설정한 값을 유지한다. 한 장이 노출된 기존 작업 step 안에서 같은 `bash scripts/run.sh ...` 명령을 사용한다.

새 할당이 필요할 때만 서버 규칙에 맞는 값을 지정한다. 아직 partition/account/GRES/최대시간 정보가 없으므로 실제 허용 값을 채워야 한다.

```bash
squeue -u "$USER"
CM_PARTITION='실제_partition'
CM_GRES='gpu:a100:1'
CM_CPUS='4'
CM_MEM='서버에서_허용한_RAM_예_64G'
CM_WALLTIME='서버에서_허용한_최대시간_예_2-00:00:00'
mkdir -p orchestration/slurm_logs

sbatch --job-name=cm-branch --partition="$CM_PARTITION" --gres="$CM_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task="$CM_CPUS" --mem="$CM_MEM" --time="$CM_WALLTIME" \
  --output=orchestration/slurm_logs/branch_%j.log \
  scripts/slurm_entry.sh configs/branch_p025.json orchestration/branch_p025

sbatch --job-name=cm-matched --partition="$CM_PARTITION" --gres="$CM_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task="$CM_CPUS" --mem="$CM_MEM" --time="$CM_WALLTIME" \
  --output=orchestration/slurm_logs/matched_%j.log \
  scripts/slurm_entry.sh configs/vanilla_matched.json orchestration/vanilla_matched

sbatch --job-name=cm-original --partition="$CM_PARTITION" --gres="$CM_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task="$CM_CPUS" --mem="$CM_MEM" --time="$CM_WALLTIME" \
  --output=orchestration/slurm_logs/original_%j.log \
  scripts/slurm_entry.sh configs/vanilla_original.json orchestration/vanilla_original
```

account가 필요하면 각 제출에 `--account=실제_account`를 추가한다. 기본CPU thread는4개다. 이미16장 할당을 받은 경우 서버의 `srun`/GPU binding 정책으로 각 작업에 한 장만 보이게 step을 구성한다. 이 코드가 기존 할당을 해제하거나 바꾸지는 않는다.

같은 할당에서 후속 조건을 순차 실행하려면 큐 완료 전 append할 수 있다. 남은 시간·자원·후속 실험 권한을 먼저 확인한다:

```bash
.venv/bin/python scripts/queue.py append --root orchestration/branch_p025 --config configs/vanilla_matched.json
```

append하면 기준선의 stage로그는 branch_p025 큐 아래에 생기지만 결과 폴더는 여전히 vanilla_matched다. 같은 config를 별도 큐와 동시에 실행하지 않는다. 자동 무한 재시도/무기한 유휴 예약은 없으며 전체 완료 시 반환한다. 실패 시 기본30분 수리 대기를 지원하고, scheduler wall time·preemption·노드 장애를 넘어서 allocation을 유지하지는 못한다.

## 7. 진행 확인

```bash
.venv/bin/python scripts/queue.py status --root orchestration/branch_p025
.venv/bin/python scripts/queue.py status --root orchestration/vanilla_matched
tail -F orchestration/branch_p025/logs/branch_p025_smoke.log
tail -F orchestration/branch_p025/logs/branch_p025_train.log
```

tail은 하나씩 실행하고 Ctrl-C로 관찰을 끝낸다. 학습 프로세스는 중단되지 않는다. `collect_session`과 `session_committed` 로그의 history/phase/next_phase를 확인한다. `progress.json`은 세션을 확정할 때 갱신하므로 긴 수집 중에는 값이 그대로일 수 있다.

| 확인 시점 | 파일/정상 값 |
|---|---|
| GPU smoke | `smoke/SMOKE_COMPLETE.json`: complete, GPU명, CUDA빌드, peak_allocated_bytes, manager만 변경 |
| 단계 전환 | metrics의 phase_used=1,next_phase=2 및 end_occupancy 네 값≥0.95 |
| 학습 중 | progress.next_batch 증가, branch_groups/학습출력 수, manager 업데이트 수 |
| 학습 완료 | `TRAIN_COMPLETE.json`: complete, 기본142 completed_sessions, checkpoint와 phases |
| 평가 완료 | `evaluation/EVAL_COMPLETE.json`: complete, questions=729, pooled 지표 |
| 분석 완료 | `ANALYSIS_COMPLETE.json`: complete, run 통계 |
| 전체 큐 완료 | 큐 `ALL_COMPLETE.json`, status=complete_allocation_releasing |

2단계에서 무작위 선택된 상태가0개이거나 advantage가 모두0인 세션은 optimizer.step이0이어도 정상이다. `branch_groups`, `signal_groups`, `invalid_executed`, 출력별reward/advantage를 함께 확인한다. 한 대화에서 네 경로가 끝까지 동시에95%를 넘지 않으면 그 대화는 전부1단계로 남을 수 있다.

## 8. 결과 비교와 반환 파일

```bash
.venv/bin/python -m cmbranch.analyze \
  --run runs/branch_v1/branch_p025 \
  --run runs/branch_v1/vanilla_matched \
  --run runs/branch_v1/vanilla_original \
  --report analysis/comparison.json
```

이 명령은 CPU로 저장 결과를 읽으며 학습을 시작하지 않는다. 미완료 조건은 incomplete/training_complete로 표시한다. 출력 coverage는0~1 비율이다. 비교 JSON에는 다음이 들어 있다:

- 공식 평가의 pooled `complete`(근거 커버리지), recall, hit와 실제 REPLACE의 근거 확보/상실 event.
- 95% 미만의 근거/비근거 턴 ADD 수·분모·비율.
- 단계별 세션 수·분기 수·생성/학습 출력 및 토큰 수·수집/업데이트 시간.
- 대화/경로별95% 최초 도달 및 전환 시점·세션 간격.

학습 세부 기록은 `batches/batch_XXXXXX/metrics.json`과 `traces.json`이다. traces의 `streams[].branch_groups[].rows[]`에 K개 후보의 원래 출력·reward·raw_advantage·정규화 advantage가 있다. `streams[].attempts[]`에는1단계 재시도 또는2단계 비분기 출력을 보존하며 `used_for_training`으로 구별한다. 학습 모니터의 REPLACE 손실은 **채점된 분기에서 실제 실행한 경우만**이며 평가 후처리는 모든 실제 REPLACE를 다룬다.

박사님께 우선 반환받을 파일:

1. 각 실험의 `smoke/SMOKE_COMPLETE.json`, `TRAIN_COMPLETE.json`, `evaluation/EVAL_COMPLETE.json`, `ANALYSIS_COMPLETE.json`.
2. `analysis/comparison.json`, 각 큐 `state.json`/`queue.json`/`logs/`/`ALL_COMPLETE.json`.
3. `validation/preflight_*.json`, `installed.freeze.txt`, GPU/드라이버 정보, `git rev-parse HEAD`와 `git status --short`.
4. 최종 checkpoint 경로 전체와 배치별trace/metrics 보존. LoRA 파일만으로 optimizer/RNG/뱅크/phase를 복구할 수 없다.

QA reader를 실행하지 않으므로 EM/F1·답변 정답률은 나오지 않는다. 첨부 시뮬 숫자나 과거 vLLM 성능과 직접 연결하지 않고 같은 엔진의 제공 기준선과 비교한다. raw데이터·가중치·긴 trace를 Git에 추가하지 않는다.

## 9. 중단·재개·오류

```bash
# 실제 stage ID는 status로 확인. 학습 단계만 중단/재개하는 예시.
.venv/bin/python scripts/queue.py cancel --root orchestration/branch_p025 --id branch_p025_train
.venv/bin/python scripts/queue.py resume --root orchestration/branch_p025 --id branch_p025_train
```

smoke/train/eval/analyze 각각 별도 stage다. 실패한 stage의 로그를 확인하고 해당 ID만 resume한다. supervisor가 살아 있으면 run 명령을 중복 시작하지 않는다. 이미 종료되었으면 기존 allocation을 먼저 확인한 뒤 같은 run 명령과 resume 요청으로 이어간다. 학습은 마지막 COMMITTED 세션 다음부터 재개하고 중간에 끊긴 세션은 재수집한다. 평가에는 턴별 진행 저장이 있다.

| 오류 | 확인 사항 |
|---|---|
| GitHub Repository not found | 비공개 repo 초대/인증. ZIP 실행도 가능 |
| 입력 SHA mismatch | 새 ZIP/파일 확인, 기존 변경 파일을 덮어쓰지 말고 새 폴더 사용 |
| version mismatch | Python3.10, `.venv`, 고정버전 확인 |
| Expose exactly one GPU | GPU mask 또는 Slurm step 확인 |
| CUDA OOM/driver 오류 | 실패로그·GPU점유·smoke peak 보존, 서버 드라이버/wheel 확인 |
| previous stage remains alive | 기록된 owner/child를 확인, 전체 pkill이나 중복실행 금지 |
| changed fingerprint | 같은 source/config/runtime인지 확인; 새 조건이면 새 checkout/출력 폴더 |

OOM 해결을 위해 K/p/질문풀/예산/정밀도를 자동 변경하지 않는다. 연구 조건을 바꿀 때는 새 실험으로 구분한다. GPU 실행 중 source를 수정해도 이미 로드된 Python은 바뀌지 않는다. 기본 수리대기는30분이며 시간이 지나면 실패 상태로 자원을 반환한다.
