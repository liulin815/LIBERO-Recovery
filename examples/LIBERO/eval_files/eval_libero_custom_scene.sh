#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Custom-scene eval launcher (independent of all other eval scripts).
#
# Counterpart to eval_libero_custom_bddl.sh, but drives eval_libero_custom_scene.py:
#   - each scene (bddl + sibling *_sim_state.npy) is evaluated for
#     num_close_trials + num_open_trials rollouts (default 5 + 5 = 10),
#   - every rollout applies a fresh small random xy perturbation to object positions,
#   - half the rollouts start with the gripper CLOSED, half OPEN.
#
# It does NOT touch any shared port range / log path used by the other launchers.
# =============================================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false

###################################
# User-configurable parameters
###################################
CONDA_SH="/H20_vepfs/liulin/miniconda3/etc/profile.d/conda.sh"
SERVER_ENV="starvla"
CLIENT_ENV="libero"
source "${CONDA_SH}"

STARVLA_PATH="/H20_vepfs/liulin/starVLA"
LIBERO_HOME="/H20_vepfs/liulin/starVLA/LIBERO"

SCENE_DIR="/H20_vepfs/liulin/starVLA/assets/scenes/selected_bddl_scenes_spatial"
LOG_PATH="/H20_vepfs/liulin/starVLA/LIBERO/eval_result/custom_scene"
GPUS_CSV="0,1"
SERVER_WORKS_PER_MODEL=2
BASE_PORT=9885          # independent of the other launchers (9883/9884)
SEED=7
RETRY_TIMES=3
NUM_STEPS_WAIT=0
NUM_SCENES=30            # randomly sample this many scenes per run (<=0 = use all)
NUM_CLOSE_TRIALS=5
NUM_OPEN_TRIALS=5
OBJ_NOISE_XY=0.02
GRIPPER_SETTLE_STEPS=20
SERVER_WARMUP_SECONDS=60
SERVER_READY_TIMEOUT_SECONDS=1800
EVAL_TIMEOUT_SECONDS=0

declare -A CHECKPOINTS_PATH=(
  #["Qwen3-VL-PI-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
  #["Qwen2.5-VL-FAST-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/Qwen2.5-VL-FAST-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
  #["Qwen2.5-VL-GR00T-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
  #["Qwen3-VL-OFT-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
  #["WM4A-Wan2d2-OFT-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt"
  #["WM4A-CosmoPredict-GR00T-LIBERO-4in1"]="/H20_vepfs/liulin/starVLA/VLA_models/WM4A-CosmoPredict-GR00T-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
  ["our_refined_model"]="/H20_vepfs/liulin/starVLA/results_custom/Checkpoints/qwen3oft_baseline_mixtraining/checkpoints/steps_50000_pytorch_model.pt"
)

###################################

C_RESET='\033[0m'
C_RED='\033[31m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_BLUE='\033[34m'
C_CYAN='\033[36m'
C_MAGENTA='\033[35m'
C_BOLD='\033[1m'

usage() {
  cat <<EOF
Usage:
  bash $(basename "$0") \\
    [--algorithms all|a1,a2,...] \\
    [--scene_dir /path/to/scenes] \\
    [--gpus 0,1] \\
    [--base_port 9885] \\
    [--seed 7] \\
    [--retry_times 3] \\
    [--num_steps_wait 0] \\
    [--num_scenes 30] \\
    [--num_close_trials 5] \\
    [--num_open_trials 5] \\
    [--obj_noise_xy 0.02] \\
    [--gripper_settle_steps 20] \\
    [--server_works_per_model 2] \\
    [--server_warmup_seconds 60] \\
    [--server_ready_timeout_seconds 1800] \\
    [--eval_timeout_seconds 0] \\
    [--log_path ${LOG_PATH}]
EOF
}

split_csv() {
  local csv="$1"
  local -n out_arr_ref=$2
  IFS=',' read -r -a out_arr_ref <<< "$csv"
}

ALGOS_ARG="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithms) ALGOS_ARG="$2"; shift 2 ;;
    --scene_dir) SCENE_DIR="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --base_port) BASE_PORT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --retry_times) RETRY_TIMES="$2"; shift 2 ;;
    --num_steps_wait) NUM_STEPS_WAIT="$2"; shift 2 ;;
    --num_scenes) NUM_SCENES="$2"; shift 2 ;;
    --num_close_trials) NUM_CLOSE_TRIALS="$2"; shift 2 ;;
    --num_open_trials) NUM_OPEN_TRIALS="$2"; shift 2 ;;
    --obj_noise_xy) OBJ_NOISE_XY="$2"; shift 2 ;;
    --gripper_settle_steps) GRIPPER_SETTLE_STEPS="$2"; shift 2 ;;
    --server_works_per_model) SERVER_WORKS_PER_MODEL="$2"; shift 2 ;;
    --server_warmup_seconds) SERVER_WARMUP_SECONDS="$2"; shift 2 ;;
    --server_ready_timeout_seconds) SERVER_READY_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --eval_timeout_seconds) EVAL_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --log_path) LOG_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo -e "${C_RED}[ERROR] Unknown arg: $1${C_RESET}"; usage; exit 1 ;;
  esac
done

HAS_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then
  HAS_TIMEOUT=1
fi

split_csv "${GPUS_CSV}" GPUS
NUM_GPUS=${#GPUS[@]}
[[ ${NUM_GPUS} -gt 0 ]] || { echo -e "${C_RED}[ERROR] No GPU selected${C_RESET}"; exit 1; }
[[ ${SERVER_WORKS_PER_MODEL} -gt 0 ]] || { echo -e "${C_RED}[ERROR] server_works_per_model must be > 0${C_RESET}"; exit 1; }
[[ -d "${SCENE_DIR}" ]] || { echo -e "${C_RED}[ERROR] scene_dir does not exist: ${SCENE_DIR}${C_RESET}"; exit 1; }

SELECTED_ALGOS=()
if [[ "${ALGOS_ARG}" == "all" ]]; then
  for k in "${!CHECKPOINTS_PATH[@]}"; do SELECTED_ALGOS+=("$k"); done
else
  split_csv "${ALGOS_ARG}" CAND_ALGOS
  for a in "${CAND_ALGOS[@]}"; do
    if [[ -n "${CHECKPOINTS_PATH[$a]:-}" ]]; then
      SELECTED_ALGOS+=("$a")
    else
      echo -e "${C_YELLOW}[WARN] Unknown algorithm skipped: $a${C_RESET}"
    fi
  done
fi

[[ ${#SELECTED_ALGOS[@]} -gt 0 ]] || { echo -e "${C_RED}[ERROR] No valid algorithm selected${C_RESET}"; exit 1; }

TOTAL_JOBS=${#SELECTED_ALGOS[@]}
TOTAL_SERVERS=$(( ${#SELECTED_ALGOS[@]} * SERVER_WORKS_PER_MODEL ))

export LIBERO_HOME="${LIBERO_HOME}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_PATH}:${PYTHONPATH:-}"

mkdir -p "${LOG_PATH}"
TMP_ROOT="${LOG_PATH}/.tmp_eval_custom_scene_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TMP_ROOT}"
PROGRESS_FILE="${TMP_ROOT}/progress.tsv"
: > "${PROGRESS_FILE}"
MASTER_LOG="${TMP_ROOT}/eval_custom_scene.log"
exec > >(stdbuf -oL -eL tee -a "${MASTER_LOG}") 2>&1

declare -A SERVER_GPU
declare -A SERVER_PORT
declare -A SERVER_PID
declare -A SERVER_MODEL
declare -A MODEL_SERVERS_CSV

SERVER_IDS=()
SERVER_PIDS=()
RUN_PORTS=()
JOB_PIDS=()
MONITOR_PID=""

cleanup() {
  set +e
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" >/dev/null 2>&1
  fi
  for pid in "${JOB_PIDS[@]:-}"; do
    kill -0 "${pid}" >/dev/null 2>&1 && kill "${pid}" >/dev/null 2>&1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill -0 "${pid}" >/dev/null 2>&1 && kill "${pid}" >/dev/null 2>&1
  done
}
trap cleanup EXIT

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn "sport = :${port}" | grep -q . && return 0
    return 1
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1 && return 0
    return 1
  fi
  return 1
}

find_free_port() {
  local preferred="$1"
  local p="${preferred}"
  while [[ ${p} -le 65535 ]]; do
    local used_in_run=0
    local rp
    for rp in "${RUN_PORTS[@]:-}"; do
      [[ "${rp}" == "${p}" ]] && used_in_run=1 && break
    done
    if [[ ${used_in_run} -eq 0 ]] && ! port_in_use "${p}"; then
      echo "${p}"; return 0
    fi
    p=$((p + 1))
  done
  return 1
}

wait_server_ready() {
  local sid="$1"
  local timeout_s="$2"
  local port="${SERVER_PORT[$sid]}"
  local pid="${SERVER_PID[$sid]}"
  local log_file="${TMP_ROOT}/server_${sid}.log"
  local start_ts now_ts
  start_ts=$(date +%s)
  while true; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo -e "${C_RED}[ERROR] server exited before ready sid=${sid}${C_RESET}"
      tail -n 100 "${log_file}" || true
      return 1
    fi
    if grep -qE "server listening on .*:${port}" "${log_file}" 2>/dev/null; then
      echo -e "${C_GREEN}[READY] sid=${sid}, port=${port}${C_RESET}"
      return 0
    fi
    if port_in_use "${port}"; then
      echo -e "${C_GREEN}[READY] sid=${sid}, port=${port}${C_RESET}"
      return 0
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      echo -e "${C_RED}[ERROR] server ready timeout sid=${sid}, port=${port}${C_RESET}"
      tail -n 100 "${log_file}" || true
      return 1
    fi
    sleep 1
  done
}

start_one_server() {
  local algo="$1"
  local model_server_idx="$2"
  local gpu="$3"
  local ckpt="$4"
  local idx="$5"
  local sid="${algo}__s${model_server_idx}"
  local preferred_port=$((BASE_PORT + idx))
  local port
  port=$(find_free_port "${preferred_port}")
  [[ -n "${port}" ]] || { echo -e "${C_RED}[ERROR] No free port for ${sid}${C_RESET}"; exit 1; }

  local server_log="${TMP_ROOT}/server_${sid}.log"
  SERVER_GPU["${sid}"]="${gpu}"
  SERVER_PORT["${sid}"]="${port}"
  SERVER_MODEL["${sid}"]="${algo}"
  SERVER_IDS+=("${sid}")
  RUN_PORTS+=("${port}")

  if [[ -n "${MODEL_SERVERS_CSV[$algo]:-}" ]]; then
    MODEL_SERVERS_CSV["$algo"]+="|${sid}"
  else
    MODEL_SERVERS_CSV["$algo"]="${sid}"
  fi

  (
    cd "${STARVLA_PATH}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    conda run --no-capture-output -n "${SERVER_ENV}" python deployment/model_server/server_policy.py \
      --ckpt_path "${ckpt}" \
      --port "${port}" \
      --use_bf16 2>&1 | stdbuf -oL -eL tee -a "${server_log}" >/dev/null
  ) &
  SERVER_PID["${sid}"]="$!"
  SERVER_PIDS+=("$!")

  echo -e "${C_GREEN}[SERVER] sid=${sid} algo=${algo} gpu=${gpu} port=${port} pid=$!${C_RESET}"
}

echo -e "${C_BOLD}${C_CYAN}========== Custom Scene Eval (perturb + gripper split) ==========${C_RESET}"
echo -e "${C_BLUE}[INFO] GPUs=${GPUS_CSV}, servers/model=${SERVER_WORKS_PER_MODEL}${C_RESET}"
echo -e "${C_BLUE}[INFO] algorithms=${#SELECTED_ALGOS[@]}, scene_dir=${SCENE_DIR}${C_RESET}"
echo -e "${C_BLUE}[INFO] trials/scene=$((NUM_CLOSE_TRIALS + NUM_OPEN_TRIALS)) "
echo -e "       (${NUM_CLOSE_TRIALS} close / ${NUM_OPEN_TRIALS} open), obj_noise_xy=${OBJ_NOISE_XY}, gripper_settle=${GRIPPER_SETTLE_STEPS}${C_RESET}"
echo -e "${C_BLUE}[INFO] total_jobs=${TOTAL_JOBS}, total_servers=${TOTAL_SERVERS}${C_RESET}"

global_server_idx=0
for algo in "${SELECTED_ALGOS[@]}"; do
  ckpt="${CHECKPOINTS_PATH[$algo]}"
  [[ -f "${ckpt}" ]] || { echo -e "${C_RED}[ERROR] Checkpoint not found: ${ckpt}${C_RESET}"; exit 1; }
  for ((sidx=1; sidx<=SERVER_WORKS_PER_MODEL; sidx++)); do
    gpu="${GPUS[$((global_server_idx % NUM_GPUS))]}"
    start_one_server "${algo}" "${sidx}" "${gpu}" "${ckpt}" "${global_server_idx}"
    global_server_idx=$((global_server_idx + 1))
  done
done

echo -e "${C_MAGENTA}[INFO] Waiting ${SERVER_WARMUP_SECONDS}s for warmup...${C_RESET}"
sleep "${SERVER_WARMUP_SECONDS}"
for sid in "${SERVER_IDS[@]}"; do
  wait_server_ready "${sid}" "${SERVER_READY_TIMEOUT_SECONDS}"
done

run_one_eval() {
  local sid="$1"
  local algo="$2"
  local gpu_id="$3"
  local port="${SERVER_PORT[$sid]}"
  local ckpt="${CHECKPOINTS_PATH[$algo]}"
  local out_dir="${LOG_PATH}/${algo}/custom_scene"
  local run_log="${TMP_ROOT}/eval_${algo}_custom_scene.log"
  mkdir -p "${out_dir}/logs"

  local attempt=0
  local ok=0
  local score="nan"
  while [[ ${attempt} -le ${RETRY_TIMES} ]]; do
    attempt=$((attempt + 1))
    set +e
    (
      cd "${STARVLA_PATH}"
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export PYTHONUNBUFFERED=1

      if [[ "${EVAL_TIMEOUT_SECONDS}" -gt 0 && "${HAS_TIMEOUT}" -eq 1 ]]; then
        timeout --signal=TERM --kill-after=30 "${EVAL_TIMEOUT_SECONDS}" \
          conda run --no-capture-output -n "${CLIENT_ENV}" \
            python ./examples/LIBERO/eval_files/eval_libero_custom_scene.py \
              --args.pretrained-path "${ckpt}" \
              --args.host "127.0.0.1" \
              --args.port "${port}" \
              --args.scene-dir "${SCENE_DIR}" \
              --args.video-out-path "${out_dir}" \
              --args.seed "${SEED}" \
              --args.num-steps-wait "${NUM_STEPS_WAIT}" \
              --args.num-scenes "${NUM_SCENES}" \
              --args.num-close-trials "${NUM_CLOSE_TRIALS}" \
              --args.num-open-trials "${NUM_OPEN_TRIALS}" \
              --args.obj-noise-xy "${OBJ_NOISE_XY}" \
              --args.gripper-settle-steps "${GRIPPER_SETTLE_STEPS}" \
              --args.job-name "${algo}_custom_scene" \
              --args.save-dataset "True" \
              --args.dataset-out-path "${out_dir}/dataset"
      else
        conda run --no-capture-output -n "${CLIENT_ENV}" \
          python ./examples/LIBERO/eval_files/eval_libero_custom_scene.py \
            --args.pretrained-path "${ckpt}" \
            --args.host "127.0.0.1" \
            --args.port "${port}" \
            --args.scene-dir "${SCENE_DIR}" \
            --args.video-out-path "${out_dir}" \
            --args.seed "${SEED}" \
            --args.num-steps-wait "${NUM_STEPS_WAIT}" \
            --args.num-scenes "${NUM_SCENES}" \
            --args.num-close-trials "${NUM_CLOSE_TRIALS}" \
            --args.num-open-trials "${NUM_OPEN_TRIALS}" \
            --args.obj-noise-xy "${OBJ_NOISE_XY}" \
            --args.gripper-settle-steps "${GRIPPER_SETTLE_STEPS}" \
            --args.job-name "${algo}_custom_scene" \
            --args.save-dataset "True" \
            --args.dataset-out-path "${out_dir}/dataset"
      fi
    ) 2>&1 | stdbuf -oL -eL tee -a "${run_log}" >/dev/null
    local exit_code=$?
    set -e
    if [[ ${exit_code} -ne 0 ]]; then
      echo -e "${C_YELLOW}[WARN] Eval failed ${algo} custom_scene try=${attempt}${C_RESET}"
      continue
    fi

    score=$(grep -E "Current total success rate:" "${run_log}" | tail -n 1 | awk -F': ' '{print $2}' || true)
    [[ -n "${score}" ]] || score="nan"
    ok=1
    break
  done

  if [[ ${ok} -eq 1 ]]; then
    printf "%s\t%s\t%s\tOK\t%s\n" "${algo}" "${sid}" "${gpu_id}" "${score}" >> "${PROGRESS_FILE}"
    echo -e "${C_GREEN}[OK]${C_RESET} ${algo} | sid=${sid} gpu=${gpu_id} | score=${score}"
  else
    printf "%s\t%s\t%s\tFAIL\tnan\n" "${algo}" "${sid}" "${gpu_id}" >> "${PROGRESS_FILE}"
    echo -e "${C_RED}[FAIL]${C_RESET} ${algo} | sid=${sid}"
  fi
}

monitor_progress() {
  local total="$1"
  local last_done=-1
  while true; do
    local done_count
    done_count=$(wc -l < "${PROGRESS_FILE}" 2>/dev/null || echo 0)
    [[ ${done_count} -gt ${total} ]] && done_count=${total}
    if [[ ${done_count} -ne ${last_done} ]]; then
      last_done=${done_count}
      echo -e "${C_CYAN}[PROGRESS]${C_RESET} done=${done_count}/${total}"
    fi
    [[ ${done_count} -ge ${total} ]] && break
    sleep 2
  done
}

echo -e "${C_MAGENTA}[INFO] Running eval jobs...${C_RESET}"
monitor_progress "${TOTAL_JOBS}" &
MONITOR_PID=$!

for algo in "${SELECTED_ALGOS[@]}"; do
  IFS='|' read -r -a algo_servers <<< "${MODEL_SERVERS_CSV[$algo]}"
  sid="${algo_servers[0]}"
  gpu_id="${SERVER_GPU[$sid]}"
  run_one_eval "${sid}" "${algo}" "${gpu_id}" &
  JOB_PIDS+=("$!")
done

for pid in "${JOB_PIDS[@]}"; do
  set +e
  wait "${pid}"
  set -e
done

if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  wait "${MONITOR_PID}" || true
fi

echo -e "${C_MAGENTA}[INFO] Aggregating results...${C_RESET}"
SELECTED_ALGOS_CSV="$(IFS=,; echo "${SELECTED_ALGOS[*]}")"

python - <<PY
import os
import math
from collections import defaultdict

progress_file = r"${PROGRESS_FILE}"
log_path = r"${LOG_PATH}"
selected_algos = [x for x in r"${SELECTED_ALGOS_CSV}".split(",") if x]

rows = []
with open(progress_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        algo, sid, gpu, status, score = line.split("\t")
        try:
            s = float(score)
        except Exception:
            s = math.nan
        rows.append({"algo": algo, "sid": sid, "gpu": gpu, "status": status, "score": s})

summary_lines = []
summary_lines.append("Custom Scene eval summary")
summary_lines.append(f"total_jobs: {len(selected_algos)}")
summary_lines.append(f"recorded_jobs: {len(rows)}")

for r in rows:
    algo = r["algo"]
    score_str = f"{r['score']:.6f}" if not math.isnan(r["score"]) else "nan"
    summary_lines.append(f"algo={algo}, status={r['status']}, score={score_str}")

# Also surface the per-scene JSON summary if the client wrote one.
for algo in selected_algos:
    scene_summary = os.path.join(log_path, algo, "custom_scene", "scene_summary.json")
    if os.path.isfile(scene_summary):
        summary_lines.append(f"per-scene detail ({algo}): {scene_summary}")

summary_file = os.path.join(log_path, "eval_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"Aggregation done: {summary_file}")
PY

echo -e "${C_BOLD}${C_GREEN}[DONE] All jobs finished.${C_RESET}"
echo -e "${C_GREEN}[DONE] Raw progress: ${PROGRESS_FILE}${C_RESET}"
echo -e "${C_GREEN}[DONE] Global summary: ${LOG_PATH}/eval_summary.txt${C_RESET}"
