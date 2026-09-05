#!/usr/bin/env bash
#set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

source /H20_vepfs/liulin/miniconda3/etc/profile.d/conda.sh
conda activate Robotwin
cd /H20_vepfs/liulin/RoboTwin
python script/update_embodiment_config_path.py


###################################
# User-configurable parameters
###################################
CONDA_SH="/H20_vepfs/liulin/miniconda3/etc/profile.d/conda.sh"
STARVLA_ENV="starvla"
EVAL_ENV="Robotwin"

STARVLA_PATH="/H20_vepfs/liulin/starVLA"
ROBOTWIN_PATH="/H20_vepfs/liulin/RoboTwin"
EVAL_FILES_PATH="${STARVLA_PATH}/examples/Robotwin/eval_files"

LOG_PATH="/H20_vepfs/liulin/RoboTwin/test"

#GPUS_CSV="0"
GPUS_CSV="0,1,2,3"
# 每个server同时跑多少个client任务
TASK_WORKERS_PER_SERVER=1
# 每个模型启动多少个server
SERVER_WORKS_PER_MODEL=4 # 6
# 评估随机种子
SEED=0
# 失败任务重试次数
RETRY_TIMES=3
# server启动等待秒数
SERVER_WARMUP_SECONDS=120
# server就绪检查超时（秒），用于大模型加载场景
SERVER_READY_TIMEOUT_SECONDS=6000
# 单个评估任务超时（秒），0表示不设超时
EVAL_TIMEOUT_SECONDS=0
# 是否打印详细进程快照与每个job发射日志（0更干净，1更详细）
VERBOSE_PROCESS_LOGS=0
# 每个server对应client的启动间隔（秒）
SLEEP_TIME_PER_CLIENT=5
# 评估数据保存配置
SAVE_EVAL_DATA=0
SAVE_MODE="all"
EVAL_DATA_SAVE_DIR="/H20_vepfs/liulin/starVLA/datasets/eval_output_wm"
# eval_result输出目录（用于隔离多次并行评估）
EVAL_RESULT_DIR="${ROBOTWIN_PATH}/eval_result_wm"

# 彩色日志
C_RESET='\033[0m'
C_RED='\033[31m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_BLUE='\033[34m'
C_CYAN='\033[36m'
C_MAGENTA='\033[35m'
C_BOLD='\033[1m'

# checkpoints_path = {"name":"path"}
declare -A CHECKPOINTS_PATH=(
  # ["stage2_vqvae_stage3_qwen3OFT_chunk50_CoT_ActionDecoder_oneview"]="/e2e-data/evad-tech-vla/zhoulei21/LAM_Project/stage3/stage2to3_actionhead/window/robotwin/0416_robotwin_lerobot_mix_50_oneview_stage2_vqvae_stage3_qwen3OFT_chunk50_CoT_ActionDecoder_oneview/checkpoints/steps_150000_pytorch_model.pt"
  # ["stage2_vqvae_stage3_qwen3OFT_chunk50_CoT_Replace_oneview"]="/e2e-data/evad-tech-vla/zhoulei21/LAM_Project/stage3/stage2to3_actionhead/window/robotwin/0416_robotwin_lerobot_mix_50_oneview_stage2_vqvae_stage3_qwen3OFT_chunk50_CoT_Replace_oneview/checkpoints/steps_150000_pytorch_model.pt"
  # ["stage2_ae_stage3_qwen3OFT_chunk50_CoT_ActionDecoder_oneview"]="/e2e-data/evad-tech-vla/zhoulei21/LAM_Project/stage3/stage2to3_actionhead/window/robotwin/0416_robotwin_lerobot_mix_50_oneview_stage2_ae_stage3_qwen3OFT_chunk50_CoT_ActionDecoder_oneview/checkpoints/steps_150000_pytorch_model.pt"
  # ["stage2_ae_stage3_qwen3OFT_chunk50_CoT_Replace_oneview"]="/e2e-data/evad-tech-vla/zhoulei21/LAM_Project/stage3/stage2to3_actionhead/window/robotwin/0416_robotwin_lerobot_mix_50_oneview_stage2_ae_stage3_qwen3OFT_chunk50_CoT_Replace_oneview/checkpoints/steps_150000_pytorch_model.pt"
  ["levla"]="/H20_vepfs/liulin/starVLA/VLA_models/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
)
# task_type = {"demo_clean","demo_randomized"}
ALL_TASK_TYPES=("demo_clean")

# task_name = {50 tasks}
# ALL_TASKS=(
#   "lift_pot"
#   "place_container_plate"
#   "move_playingcard_away"
#   "adjust_bottle"
#   "press_stapler"
#   "place_object_stand"
#   "move_stapler_pad"
#   "rotate_qrcode"
#   "place_mouse_pad"
#   "turn_switch"
#   "place_phone_stand"
#   "pick_diverse_bottles"
# )
ALL_TASKS=(
    "stack_blocks_three"
    "place_bread_basket"
    "put_bottles_dustbin"
    "place_bread_skillet"
    "pick_diverse_bottles"
    "blocks_ranking_size"
    "place_can_basket"
    "turn_switch"
    "click_alarmclock"
    "hanging_mug"
    "open_microwave"
    "click_bell"
)
# ALL_TASKS=(
#       "hanging_mug"
#       "click_alarmclock"
#       "click_bell"
# )
###################################

usage() {
  cat <<EOF
Usage:
  bash eval_robotwin_multigpu.sh \
    [--task_name all|t1,t2,...] \
    [--task_type all|demo_clean,demo_randomized] \
    [--algorithms all|a1,a2,...] \
    [--task_workers_per_server 4] \
    [--server_works_per_model 1] \
    [--sleep_time_per_client 5] \
    [--gpus 0,1,2,3,4,5,6,7] \
    [--seed 0] \
    [--log_path ${LOG_PATH}] \
    [--retry_times 1] \
    [--server_warmup_seconds 25] \
    [--server_ready_timeout_seconds 600] \
    [--eval_timeout_seconds 0] \
    [--save_eval_data] \
    [--save_mode success|fail|all] \
    [--eval_data_save_dir /path/to/save]
EOF
}

split_csv() {
  local csv="$1"
  local -n out_arr_ref=$2
  IFS=',' read -r -a out_arr_ref <<< "$csv"
}

contains() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

TASK_NAME_ARG="all"
TASK_TYPE_ARG="all"
ALGORITHMS_ARG="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task_name) TASK_NAME_ARG="$2"; shift 2 ;;
    --task_type) TASK_TYPE_ARG="$2"; shift 2 ;;
    --algorithms) ALGORITHMS_ARG="$2"; shift 2 ;;
    --task_workers_per_server) TASK_WORKERS_PER_SERVER="$2"; shift 2 ;;
    --server_works_per_model) SERVER_WORKS_PER_MODEL="$2"; shift 2 ;;
    --sleep_time_per_client) SLEEP_TIME_PER_CLIENT="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log_path) LOG_PATH="$2"; shift 2 ;;
    --retry_times) RETRY_TIMES="$2"; shift 2 ;;
    --server_warmup_seconds) SERVER_WARMUP_SECONDS="$2"; shift 2 ;;
    --server_ready_timeout_seconds) SERVER_READY_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --eval_timeout_seconds) EVAL_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --verbose_process_logs) VERBOSE_PROCESS_LOGS="$2"; shift 2 ;;
    --save_eval_data) SAVE_EVAL_DATA=1; shift 1 ;;
    --save_mode) SAVE_MODE="$2"; shift 2 ;;
    --eval_data_save_dir) EVAL_DATA_SAVE_DIR="$2"; shift 2 ;;
    --eval_result_dir) EVAL_RESULT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

source "${CONDA_SH}"

HAS_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then
  HAS_TIMEOUT=1
fi

split_csv "${GPUS_CSV}" GPUS
NUM_GPUS=${#GPUS[@]}
if [[ ${NUM_GPUS} -le 0 ]]; then
  echo "[ERROR] No GPU selected."; exit 1
fi

if [[ ${TASK_WORKERS_PER_SERVER} -le 0 ]]; then
  echo "[ERROR] task_workers_per_server must be > 0"; exit 1
fi
if [[ ${SERVER_WORKS_PER_MODEL} -le 0 ]]; then
  echo "[ERROR] server_works_per_model must be > 0"; exit 1
fi
if [[ ${SLEEP_TIME_PER_CLIENT} -lt 0 ]]; then
  echo "[ERROR] sleep_time_per_client must be >= 0"; exit 1
fi

# Select task list
SELECTED_TASKS=()
if [[ "${TASK_NAME_ARG}" == "all" ]]; then
  SELECTED_TASKS=("${ALL_TASKS[@]}")
else
  split_csv "${TASK_NAME_ARG}" CAND_TASKS
  for t in "${CAND_TASKS[@]}"; do
    if contains "$t" "${ALL_TASKS[@]}"; then
      SELECTED_TASKS+=("$t")
    else
      echo "[WARN] Unknown task skipped: $t"
    fi
  done
fi

# Select task type
SELECTED_TASK_TYPES=()
if [[ "${TASK_TYPE_ARG}" == "all" ]]; then
  SELECTED_TASK_TYPES=("${ALL_TASK_TYPES[@]}")
else
  split_csv "${TASK_TYPE_ARG}" CAND_TYPES
  for tt in "${CAND_TYPES[@]}"; do
    if contains "$tt" "${ALL_TASK_TYPES[@]}"; then
      SELECTED_TASK_TYPES+=("$tt")
    else
      echo "[WARN] Unknown task_type skipped: $tt"
    fi
  done
fi

# Select algorithms
SELECTED_ALGOS=()
if [[ "${ALGORITHMS_ARG}" == "all" ]]; then
  for k in "${!CHECKPOINTS_PATH[@]}"; do
    SELECTED_ALGOS+=("$k")
  done
else
  split_csv "${ALGORITHMS_ARG}" CAND_ALGOS
  for a in "${CAND_ALGOS[@]}"; do
    if [[ -n "${CHECKPOINTS_PATH[$a]:-}" ]]; then
      SELECTED_ALGOS+=("$a")
    else
      echo "[WARN] Unknown algorithm skipped: $a"
    fi
  done
fi

if [[ ${#SELECTED_TASKS[@]} -eq 0 ]]; then echo "[ERROR] No valid task selected."; exit 1; fi
if [[ ${#SELECTED_TASK_TYPES[@]} -eq 0 ]]; then echo "[ERROR] No valid task_type selected."; exit 1; fi
if [[ ${#SELECTED_ALGOS[@]} -eq 0 ]]; then echo "[ERROR] No valid algorithm selected."; exit 1; fi

TOTAL_JOBS=$(( ${#SELECTED_ALGOS[@]} * ${#SELECTED_TASK_TYPES[@]} * ${#SELECTED_TASKS[@]} ))
TOTAL_SERVERS=$(( ${#SELECTED_ALGOS[@]} * SERVER_WORKS_PER_MODEL ))
MAX_PARALLEL=$(( TOTAL_SERVERS * TASK_WORKERS_PER_SERVER ))

mkdir -p "${LOG_PATH}"
TMP_ROOT="${LOG_PATH}/.tmp_eval_robotwin_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TMP_ROOT}"
PROGRESS_FILE="${TMP_ROOT}/progress.tsv"
: > "${PROGRESS_FILE}"
MASTER_LOG_FILE="${TMP_ROOT}/eval_robotwin_multigpu.log"

# 主日志实时写入（同时输出到终端）
exec > >(stdbuf -oL -eL tee -a "${MASTER_LOG_FILE}") 2>&1

declare -A SERVER_CFG
declare -A SERVER_GPU
declare -A SERVER_PORT
declare -A SERVER_PID
declare -A SERVER_MODEL
declare -A SERVER_MODEL_IDX
SERVER_IDS=()

declare -A MODEL_SERVERS_CSV

declare -A SERVER_QUEUE_FILE
declare -A SERVER_QUEUE_POS
declare -A SERVER_QUEUE_LEN
declare -A SERVER_RUNNING
declare -A SERVER_LAST_LAUNCH_TS

declare -A PID_SERVER_ID
ACTIVE_JOB_PIDS=()
RUN_PORTS=()
SERVER_PIDS=()
MONITOR_PID=""
JOB_PIDS=()

cleanup() {
  set +e
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
    kill "${MONITOR_PID}" >/dev/null 2>&1
  fi

  for pid in "${JOB_PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1
    fi
  done

  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1
    fi
  done
}
trap cleanup EXIT

log_process_snapshot() {
  local tag="$1"
  if [[ "${VERBOSE_PROCESS_LOGS}" != "1" ]]; then
    return 0
  fi
  echo -e "${C_CYAN}[PROC] snapshot=${tag}${C_RESET}"
  set +e
  local server_cnt client_cnt
  server_cnt=$(pgrep -af "deployment/model_server/server_policy.py" | wc -l)
  client_cnt=$(pgrep -af "script/eval_policy.py" | wc -l)
  echo -e "${C_CYAN}[PROC] server_policy.py count=${server_cnt}, eval_policy_nips.py count=${client_cnt}${C_RESET}"
  pgrep -af "deployment/model_server/server_policy.py" | head -n 20 || true
  pgrep -af "script/eval_policy.py" | head -n 20 || true
  set -e
}

echo -e "${C_BOLD}${C_CYAN}========== Robotwin Multi-GPU Eval ==========${C_RESET}"
echo -e "${C_BLUE}[INFO] GPUs: ${GPUS_CSV}${C_RESET}"
echo -e "${C_BLUE}[INFO] task_workers_per_server: ${TASK_WORKERS_PER_SERVER}${C_RESET}"
echo -e "${C_BLUE}[INFO] server_works_per_model: ${SERVER_WORKS_PER_MODEL}${C_RESET}"
echo -e "${C_BLUE}[INFO] max_parallel_jobs: ${MAX_PARALLEL}${C_RESET}"
echo -e "${C_BLUE}[INFO] eval_timeout_seconds: ${EVAL_TIMEOUT_SECONDS}${C_RESET}"
echo -e "${C_BLUE}[INFO] server_ready_timeout_seconds: ${SERVER_READY_TIMEOUT_SECONDS}${C_RESET}"
echo -e "${C_BLUE}[INFO] verbose_process_logs: ${VERBOSE_PROCESS_LOGS}${C_RESET}"
echo -e "${C_BLUE}[INFO] sleep_time_per_client: ${SLEEP_TIME_PER_CLIENT}${C_RESET}"
echo -e "${C_BLUE}[INFO] total_servers: ${TOTAL_SERVERS}${C_RESET}"
echo -e "${C_BLUE}[INFO] algorithms: ${#SELECTED_ALGOS[@]}, task_types: ${#SELECTED_TASK_TYPES[@]}, tasks: ${#SELECTED_TASKS[@]}${C_RESET}"
echo -e "${C_BLUE}[INFO] total_jobs: ${TOTAL_JOBS}${C_RESET}"
echo -e "${C_BLUE}[INFO] current_tmp_log_dir: ${TMP_ROOT}${C_RESET}"

echo -e "${C_MAGENTA}[INFO] Starting policy servers...${C_RESET}"
BASE_PORT=5694
MAX_PORT=65535

port_in_use() {
  local port="$1"
  # 端口被占用(已有LISTEN)返回0；空闲返回1
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn "sport = :${port}" | grep -q . && return 0
    return 1
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1 && return 0
    return 1
  fi

  # 最后兜底：connect_ex 检测（依赖starVLA环境python）
  set +e
  conda run --no-capture-output -p "/H20_vepfs/liulin/miniconda3/envs/${STARVLA_ENV}" python -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); rc=s.connect_ex(('127.0.0.1', int(sys.argv[1]))); s.close(); sys.exit(0 if rc==0 else 1)" "${port}"
  local rc=$?
  set -e
  [[ ${rc} -eq 0 ]] && return 0
  return 1
}

find_free_port() {
  local preferred="$1"
  local p="${preferred}"

  while [[ ${p} -le ${MAX_PORT} ]]; do
    local used_in_run=0
    local rp
    for rp in "${RUN_PORTS[@]:-}"; do
      if [[ "${rp}" == "${p}" ]]; then
        used_in_run=1
        break
      fi
    done

    if [[ ${used_in_run} -eq 0 ]] && ! port_in_use "${p}"; then
      echo "${p}"
      return 0
    fi
    p=$((p + 1))
  done

  return 1
}

start_one_server() {
  local algo="$1"
  local model_server_idx="$2"
  local gpu="$3"
  local ckpt="$4"
  local idx="$5"

  local server_id="${algo}__s${model_server_idx}"

  local preferred_port=$((BASE_PORT + idx))
  local port
  if ! port=$(find_free_port "${preferred_port}"); then
    echo -e "${C_RED}[ERROR] cannot find free port in [${preferred_port}, ${MAX_PORT}] for algo=${algo}${C_RESET}"
    exit 1
  fi
  if [[ "${port}" != "${preferred_port}" ]]; then
    echo -e "${C_YELLOW}[WARN] preferred port ${preferred_port} occupied, switch to ${port} (algo=${algo})${C_RESET}"
  fi

  local cfg_file="${TMP_ROOT}/deploy_${server_id}.yml"
  local server_log="${TMP_ROOT}/server_${server_id}.log"
  cat > "${cfg_file}" <<EOF
policy_name: starVLA
task_name: null
task_config: null
ckpt_setting: null
seed: null
instruction_type: unseen
host: "127.0.0.1"
port: ${port}
policy_ckpt_path: "${ckpt}"
unnorm_key: "new_embodiment"
action_mode: "abs"
EOF
  SERVER_CFG["$server_id"]="${cfg_file}"
  SERVER_GPU["$server_id"]="${gpu}"
  SERVER_PORT["$server_id"]="${port}"
  SERVER_MODEL["$server_id"]="${algo}"
  SERVER_MODEL_IDX["$server_id"]="${model_server_idx}"
  SERVER_IDS+=("${server_id}")

  if [[ -n "${MODEL_SERVERS_CSV[$algo]:-}" ]]; then
    MODEL_SERVERS_CSV["$algo"]+="|${server_id}"
  else
    MODEL_SERVERS_CSV["$algo"]="${server_id}"
  fi

  SERVER_QUEUE_FILE["$server_id"]="${TMP_ROOT}/queue_${server_id}.tsv"
  : > "${SERVER_QUEUE_FILE[$server_id]}"
  SERVER_QUEUE_POS["$server_id"]=1
  SERVER_QUEUE_LEN["$server_id"]=0
  SERVER_RUNNING["$server_id"]=0
  SERVER_LAST_LAUNCH_TS["$server_id"]=0

  RUN_PORTS+=("${port}")

  {
    echo "[$(date '+%F %T')] launch server server_id=${server_id} algo=${algo} gpu=${gpu} port=${port}"
    echo "[$(date '+%F %T')] log_file=${server_log}"
  } >> "${server_log}"

  (
    cd "${STARVLA_PATH}"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    stdbuf -oL -eL conda run --no-capture-output -p "/H20_vepfs/liulin/miniconda3/envs/${STARVLA_ENV}" python deployment/model_server/server_policy.py \
      --ckpt_path "${ckpt}" \
      --port "${port}" \
      --use_bf16 2>&1 | stdbuf -oL -eL tee -a "${server_log}" >/dev/null
  ) &
  SERVER_PIDS+=("$!")
  SERVER_PID["$server_id"]="$!"

  echo -e "${C_GREEN}[SERVER] server_id=${server_id}, algo=${algo}, gpu=${gpu}, port=${port}, pid=$!${C_RESET}"
}

verify_algo_port_routing() {
  echo -e "${C_BLUE}[ROUTE] verifying server->port mapping...${C_RESET}"
  local sid cfg expected_port cfg_port algo gpu
  for sid in "${SERVER_IDS[@]}"; do
    cfg="${SERVER_CFG[$sid]}"
    expected_port="${SERVER_PORT[$sid]}"
    algo="${SERVER_MODEL[$sid]}"
    gpu="${SERVER_GPU[$sid]}"
    cfg_port=$(awk -F': ' '/^port:/{print $2; exit}' "${cfg}" | tr -d '[:space:]')

    if [[ -z "${cfg_port}" ]]; then
      echo -e "${C_RED}[ERROR] port not found in cfg for server_id=${sid}, cfg=${cfg}${C_RESET}"
      return 1
    fi
    if [[ "${cfg_port}" != "${expected_port}" ]]; then
      echo -e "${C_RED}[ERROR] route mismatch for server_id=${sid}: cfg_port=${cfg_port}, server_port=${expected_port}${C_RESET}"
      return 1
    fi
    echo -e "${C_GREEN}[ROUTE] server_id=${sid} algo=${algo} gpu=${gpu} -> port=${expected_port}${C_RESET}"
  done
  return 0
}

wait_server_ready() {
  local sid="$1"
  local timeout_s="$2"
  local port="${SERVER_PORT[$sid]}"
  local server_pid="${SERVER_PID[$sid]:-}"
  local server_log="${TMP_ROOT}/server_${sid}.log"
  local algo="${SERVER_MODEL[$sid]}"

  echo -e "${C_BLUE}[WAIT] checking server readiness: server_id=${sid}, algo=${algo}, port=${port}, timeout=${timeout_s}s${C_RESET}"

  local start_ts now_ts
  start_ts=$(date +%s)
  while true; do
    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo -e "${C_RED}[ERROR] server process exited before ready: server_id=${sid}, algo=${algo}, pid=${server_pid}${C_RESET}"
      tail -n 100 "${server_log}" || true
      return 1
    fi

    # 以服务端日志中的明确监听信号作为就绪条件（与手工单独启动一致）
    # 典型日志：INFO:websockets.server:server listening on 0.0.0.0:5694
    if grep -qE "server listening on .*:${port}" "${server_log}" 2>/dev/null; then
      echo -e "${C_GREEN}[READY] server ready(by log): algo=${algo}, port=${port}${C_RESET}"
      return 0
    fi

    # 兜底：端口监听也可视为就绪（防止日志格式变更）
    if port_in_use "${port}"; then
      echo -e "${C_GREEN}[READY] server ready(by port): algo=${algo}, port=${port}${C_RESET}"
      return 0
    fi

    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_s )); then
      echo -e "${C_RED}[ERROR] server not ready in time: server_id=${sid}, algo=${algo}, port=${port}${C_RESET}"
      tail -n 100 "${server_log}" || true
      return 1
    fi
    sleep 1
  done
}

# 为每个模型启动 SERVER_WORKS_PER_MODEL 个server，并按平均策略分配GPU
global_server_idx=0
for algo in "${SELECTED_ALGOS[@]}"; do
  ckpt="${CHECKPOINTS_PATH[$algo]}"
  if [[ ! -f "$ckpt" ]]; then
    echo -e "${C_RED}[ERROR] Checkpoint not found for ${algo}: ${ckpt}${C_RESET}"
    exit 1
  fi
  for ((sidx=1; sidx<=SERVER_WORKS_PER_MODEL; sidx++)); do
    gpu="${GPUS[$((global_server_idx % NUM_GPUS))]}"
    start_one_server "${algo}" "${sidx}" "${gpu}" "${ckpt}" "${global_server_idx}"
    global_server_idx=$((global_server_idx + 1))
  done
done

log_process_snapshot "after_server_start"

# wait a bit for servers
echo -e "${C_MAGENTA}[INFO] Waiting ${SERVER_WARMUP_SECONDS}s for servers warmup...${C_RESET}"
sleep "${SERVER_WARMUP_SECONDS}"

# 严格检查每个server端口可用，确保client只在server就绪后启动
for sid in "${SERVER_IDS[@]}"; do
  wait_server_ready "${sid}" "${SERVER_READY_TIMEOUT_SECONDS}"
done

verify_algo_port_routing

echo -e "${C_GREEN}[READY] all selected servers are ready. start clients now.${C_RESET}"

# 给每个模型构建任务并按server轮转分配，保证模型任务全覆盖
for algo in "${SELECTED_ALGOS[@]}"; do
  IFS='|' read -r -a algo_servers <<< "${MODEL_SERVERS_CSV[$algo]}"
  if [[ ${#algo_servers[@]} -eq 0 ]]; then
    echo -e "${C_RED}[ERROR] no servers found for model=${algo}${C_RESET}"
    exit 1
  fi

  assign_idx=0
  for task_type in "${SELECTED_TASK_TYPES[@]}"; do
    for task in "${SELECTED_TASKS[@]}"; do
      sid="${algo_servers[$((assign_idx % ${#algo_servers[@]}))]}"
      printf "%s\t%s\t%s\n" "${algo}" "${task}" "${task_type}" >> "${SERVER_QUEUE_FILE[$sid]}"
      assign_idx=$((assign_idx + 1))
    done
  done
done

for sid in "${SERVER_IDS[@]}"; do
  SERVER_QUEUE_LEN["$sid"]=$(wc -l < "${SERVER_QUEUE_FILE[$sid]}" 2>/dev/null || echo 0)
  echo -e "${C_BLUE}[QUEUE] server_id=${sid} algo=${SERVER_MODEL[$sid]} gpu=${SERVER_GPU[$sid]} tasks=${SERVER_QUEUE_LEN[$sid]}${C_RESET}"
done

run_one_eval() {
  local sid="$1"
  local algo="$2"
  local task="$3"
  local task_type="$4"
  local gpu_id="$5"

  local cfg="${SERVER_CFG[$sid]:-}"
  if [[ -z "${cfg}" ]]; then
    echo -e "${C_RED}[ERROR] no cfg found for server_id=${sid}, algo=${algo}${C_RESET}"
    printf "%s\t%s\t%s\t%s\tFAIL\t%s\t%s\n" "${algo}" "${task_type}" "${task}" "${gpu_id}" "nan" "" >> "${PROGRESS_FILE}"
    return 0
  fi

  local attempt=0
  local ok=0
  local score="nan"
  local out_txt=""

  while [[ ${attempt} -le ${RETRY_TIMES} ]]; do
    attempt=$((attempt + 1))
    local run_log="${TMP_ROOT}/eval_${algo}_${task}_${task_type}_try${attempt}.log"

    local before_count
    before_count=$(find "${EVAL_RESULT_DIR}/${task}/model2robotwin_interface/${task_type}/${algo}" -name "_result.txt" 2>/dev/null | wc -l || true)

    set +e
    (
      cd "${ROBOTWIN_PATH}"
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      # 在双显卡/混合图形环境中优先走 NVIDIA 渲染设备
      export VK_LAYER_NV_optimus=NVIDIA_only
      export __NV_PRIME_RENDER_OFFLOAD=1
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      export PYTHONPATH="${ROBOTWIN_PATH}:${STARVLA_PATH}:${EVAL_FILES_PATH}:${PYTHONPATH:-}"
      export PYTHONUNBUFFERED=1

      run_eval_cmd() {
        if [[ "${EVAL_TIMEOUT_SECONDS}" -gt 0 && "${HAS_TIMEOUT}" -eq 1 ]]; then
          timeout --signal=TERM --kill-after=30 "${EVAL_TIMEOUT_SECONDS}" "$@"
        else
          "$@"
        fi
      }

      PYTHONWARNINGS=ignore::UserWarning \
      run_eval_cmd conda run --no-capture-output -p "/H20_vepfs/liulin/miniconda3/envs/${EVAL_ENV}" python script/eval_policy.py --config "${cfg}" --overrides \
        --task_name "${task}" \
        --task_config "${task_type}" \
        --ckpt_setting "${algo}" \
        --seed "${SEED}" \
        --policy_name "model2robotwin_interface" \
        --save_eval_data "${SAVE_EVAL_DATA}" \
        --save_mode "${SAVE_MODE}" \
        --eval_data_save_dir "${EVAL_DATA_SAVE_DIR}" \
        --eval_result_dir "${EVAL_RESULT_DIR}"
    ) 2>&1 | stdbuf -oL -eL tee -a "${run_log}" >/dev/null

    local exit_code=$?
    set -e
    if [[ ${exit_code} -ne 0 ]]; then
      echo -e "${C_YELLOW}[WARN] Eval failed (exit=${exit_code}) ${algo} ${task} ${task_type} try=${attempt}${C_RESET}"
      continue
    fi

    local latest_file
    latest_file=$(find "${EVAL_RESULT_DIR}/${task}/model2robotwin_interface/${task_type}/${algo}" -name "_result.txt" 2>/dev/null | sort | tail -n 1 || true)

    if [[ -n "${latest_file}" && -f "${latest_file}" ]]; then
      out_txt="${latest_file}"
      score=$(grep -E '^[0-9]+(\.[0-9]+)?$' "${latest_file}" | tail -n 1 || true)
      if [[ -n "${score}" ]]; then
        ok=1
        break
      fi
    fi

    local after_count
    after_count=$(find "${EVAL_RESULT_DIR}/${task}/model2robotwin_interface/${task_type}/${algo}" -name "_result.txt" 2>/dev/null | wc -l || true)
    if [[ "${after_count}" -gt "${before_count}" ]]; then
      ok=1
      score="nan"
      break
    fi
  done

  if [[ ${ok} -eq 1 ]]; then
    printf "%s\t%s\t%s\t%s\tOK\t%s\t%s\n" "${algo}" "${task_type}" "${task}" "${gpu_id}" "${score}" "${out_txt}" >> "${PROGRESS_FILE}"
    echo -e "${C_GREEN}[OK]${C_RESET} ${algo} | ${task_type} | ${task} | gpu=${gpu_id} | score=${score}"
  else
    printf "%s\t%s\t%s\t%s\tFAIL\t%s\t%s\n" "${algo}" "${task_type}" "${task}" "${gpu_id}" "nan" "" >> "${PROGRESS_FILE}"
    echo -e "${C_RED}[FAIL]${C_RESET} ${algo} | ${task_type} | ${task}"
  fi

  return 0
}

monitor_progress() {
  local total="$1"

  # 非TTY输出（例如被tee到日志）时，避免\r进度条污染日志
  if [[ ! -t 1 ]]; then
    local last_done=-1
    while true; do
      local done ok fail
      done=$(wc -l < "${PROGRESS_FILE}" 2>/dev/null || echo 0)
      ok=$(awk -F'\t' 'BEGIN{c=0} $5=="OK"{c++} END{print c}' "${PROGRESS_FILE}" 2>/dev/null || echo 0)
      fail=$((done - ok))
      [[ ${done} -gt ${total} ]] && done=${total}

      if [[ ${done} -ne ${last_done} ]]; then
        echo -e "${C_CYAN}[PROGRESS]${C_RESET} done=${done}/${total} ok=${ok} fail=${fail}"
        last_done=${done}
      fi

      [[ ${done} -ge ${total} ]] && break
      sleep 2
    done
    return 0
  fi

  while true; do
    local done ok fail pct bar_w filled bar i
    done=$(wc -l < "${PROGRESS_FILE}" 2>/dev/null || echo 0)
    ok=$(awk -F'\t' 'BEGIN{c=0} $5=="OK"{c++} END{print c}' "${PROGRESS_FILE}" 2>/dev/null || echo 0)
    fail=$((done - ok))
    [[ ${done} -gt ${total} ]] && done=${total}
    pct=0
    if [[ ${total} -gt 0 ]]; then pct=$(( done * 100 / total )); fi

    bar_w=30
    filled=$(( pct * bar_w / 100 ))
    bar=""
    for ((i=0; i<filled; i++)); do bar+="#"; done
    for ((i=filled; i<bar_w; i++)); do bar+="-"; done

    printf "\r${C_CYAN}[PROGRESS]${C_RESET} [%s] %3d%% done=%d/%d ${C_GREEN}ok=%d${C_RESET} ${C_RED}fail=%d${C_RESET}" "${bar}" "${pct}" "${done}" "${total}" "${ok}" "${fail}"
    [[ ${done} -ge ${total} ]] && break
    sleep 1
  done
  echo
}

echo -e "${C_MAGENTA}[INFO] Running eval jobs...${C_RESET}"
monitor_progress "${TOTAL_JOBS}" &
MONITOR_PID=$!
launched_total=0

has_pending_jobs() {
  local sid
  for sid in "${SERVER_IDS[@]}"; do
    if (( SERVER_QUEUE_POS[$sid] <= SERVER_QUEUE_LEN[$sid] )); then
      return 0
    fi
  done
  return 1
}

reap_finished_jobs() {
  local new_active=()
  local pid sid
  for pid in "${ACTIVE_JOB_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      new_active+=("${pid}")
    else
      sid="${PID_SERVER_ID[$pid]:-}"
      set +e
      wait "${pid}" >/dev/null 2>&1
      set -e
      if [[ -n "${sid}" ]]; then
        if (( SERVER_RUNNING[$sid] > 0 )); then
          SERVER_RUNNING[$sid]=$((SERVER_RUNNING[$sid] - 1))
        fi
      fi
      unset PID_SERVER_ID["$pid"]
    fi
  done
  ACTIVE_JOB_PIDS=("${new_active[@]}")
}

launch_one_from_server() {
  local sid="$1"
  local now_ts
  now_ts=$(date +%s)

  if (( SERVER_RUNNING[$sid] >= TASK_WORKERS_PER_SERVER )); then
    return 1
  fi
  if (( SERVER_QUEUE_POS[$sid] > SERVER_QUEUE_LEN[$sid] )); then
    return 1
  fi
  if (( now_ts - SERVER_LAST_LAUNCH_TS[$sid] < SLEEP_TIME_PER_CLIENT )); then
    return 1
  fi

  local line algo task task_type gpu_id
  line=$(sed -n "${SERVER_QUEUE_POS[$sid]}p" "${SERVER_QUEUE_FILE[$sid]}")
  if [[ -z "${line}" ]]; then
    SERVER_QUEUE_POS[$sid]=$((SERVER_QUEUE_LEN[$sid] + 1))
    return 1
  fi

  IFS=$'\t' read -r algo task task_type <<< "${line}"
  gpu_id="${SERVER_GPU[$sid]}"

  run_one_eval "${sid}" "${algo}" "${task}" "${task_type}" "${gpu_id}" &
  local launched_pid="$!"
  JOB_PIDS+=("${launched_pid}")
  ACTIVE_JOB_PIDS+=("${launched_pid}")
  PID_SERVER_ID["$launched_pid"]="${sid}"
  SERVER_RUNNING[$sid]=$((SERVER_RUNNING[$sid] + 1))
  SERVER_QUEUE_POS[$sid]=$((SERVER_QUEUE_POS[$sid] + 1))
  SERVER_LAST_LAUNCH_TS[$sid]="${now_ts}"
  launched_total=$((launched_total + 1))

  if [[ "${VERBOSE_PROCESS_LOGS}" == "1" ]]; then
    echo -e "${C_BLUE}[LAUNCH] idx=${launched_total}/${TOTAL_JOBS} pid=${launched_pid} server_id=${sid} algo=${algo} task_type=${task_type} task=${task} server_running=${SERVER_RUNNING[$sid]}${C_RESET}"
  fi
  return 0
}

# 首轮：尽量保证每个server至少启动一个client
for sid in "${SERVER_IDS[@]}"; do
  launch_one_from_server "${sid}" || true
done

rr_start=0
while true; do
  reap_finished_jobs

  launched_any=0
  for ((i=0; i<${#SERVER_IDS[@]}; i++)); do
    sid="${SERVER_IDS[$(((rr_start + i) % ${#SERVER_IDS[@]}))]}"
    if launch_one_from_server "${sid}"; then
      launched_any=1
    fi
  done
  rr_start=$(((rr_start + 1) % ${#SERVER_IDS[@]}))

  if (( launched_total % 5 == 0 || launched_total == TOTAL_JOBS )); then
    log_process_snapshot "launch_progress_${launched_total}"
  fi

  if ! has_pending_jobs && [[ ${#ACTIVE_JOB_PIDS[@]} -eq 0 ]]; then
    break
  fi

  if [[ ${launched_any} -eq 0 ]]; then
    sleep 1
  fi
done

log_process_snapshot "after_all_jobs_launched"

for pid in "${JOB_PIDS[@]}"; do
  set +e
  wait "${pid}"
  set -e
done

log_process_snapshot "after_all_jobs_finished"

if kill -0 "${MONITOR_PID}" >/dev/null 2>&1; then
  kill "${MONITOR_PID}" >/dev/null 2>&1 || true
  wait "${MONITOR_PID}" || true
fi

echo -e "${C_MAGENTA}[INFO] Aggregating results...${C_RESET}"
SELECTED_TASKS_CSV="$(IFS=,; echo "${SELECTED_TASKS[*]}")"
SELECTED_TASK_TYPES_CSV="$(IFS=,; echo "${SELECTED_TASK_TYPES[*]}")"
SELECTED_ALGOS_CSV="$(IFS=,; echo "${SELECTED_ALGOS[*]}")"
python - <<PY
import os
import math
import statistics
from collections import defaultdict

progress_file = r"${PROGRESS_FILE}"
log_path = r"${LOG_PATH}"
selected_tasks = [x for x in r"${SELECTED_TASKS_CSV}".split(",") if x]
selected_task_types = [x for x in r"${SELECTED_TASK_TYPES_CSV}".split(",") if x]
selected_algos = [x for x in r"${SELECTED_ALGOS_CSV}".split(",") if x]

rows = []
with open(progress_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        algo, task_type, task, gpu, status, score, result_path = line.split("\t")
        s = None
        try:
            s = float(score)
        except Exception:
            s = math.nan
        rows.append({
            "algo": algo,
            "task_type": task_type,
            "task": task,
            "gpu": gpu,
            "status": status,
            "score": s,
            "result_path": result_path,
        })

by_algo_type = defaultdict(list)
for r in rows:
    by_algo_type[(r["algo"], r["task_type"])].append(r)

expected = {(a, tt, t) for a in selected_algos for tt in selected_task_types for t in selected_tasks}
observed = {(r["algo"], r["task_type"], r["task"]) for r in rows}
missing_triplets = sorted(expected - observed)

summary_lines = []
summary_lines.append("Robotwin eval global summary")
summary_lines.append(f"total_expected_jobs: {len(expected)}")
summary_lines.append(f"total_recorded_jobs: {len(rows)}")
summary_lines.append(f"missing_jobs: {len(missing_triplets)}")

for (algo, task_type), rs in by_algo_type.items():
    out_dir = os.path.join(log_path, algo)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{task_type}.txt")

    ok_scores = [x["score"] for x in rs if x["status"] == "OK" and not math.isnan(x["score"])]
    task_map = {x["task"]: x for x in rs}
    total = len(selected_tasks)
    ok = sum(1 for t in selected_tasks if t in task_map and task_map[t]["status"] == "OK")
    fail = total - ok
    mean = statistics.mean(ok_scores) if ok_scores else math.nan
    std = statistics.pstdev(ok_scores) if len(ok_scores) > 1 else 0.0 if len(ok_scores) == 1 else math.nan

    with open(out_file, "w", encoding="utf-8") as wf:
        wf.write(f"Algorithm: {algo}\n")
        wf.write(f"Task Type: {task_type}\n")
        wf.write(f"Total Jobs: {total}\n")
        wf.write(f"Success Jobs: {ok}\n")
        wf.write(f"Failed Jobs: {fail}\n")
        wf.write(f"Mean Success Rate: {mean:.6f}\n" if not math.isnan(mean) else "Mean Success Rate: nan\n")
        wf.write(f"Std Success Rate: {std:.6f}\n" if not math.isnan(std) else "Std Success Rate: nan\n")
        wf.write("\nPer-task results (task\tscore\tstatus\tresult_file):\n")

        missing = []
        for t in selected_tasks:
          if t in task_map:
            r = task_map[t]
            score_str = f"{r['score']:.6f}" if not math.isnan(r["score"]) else "nan"
            wf.write(f"{t}\t{score_str}\t{r['status']}\t{r['result_path']}\n")
          else:
            missing.append(t)
            wf.write(f"{t}\tnan\tMISSING\t\n")

        if missing:
            wf.write("\nMissing tasks:\n")
            for m in missing:
                wf.write(f"{m}\n")

    summary_lines.append(
        f"algo={algo}, task_type={task_type}, success={ok}/{total}, "
        f"mean={'nan' if math.isnan(mean) else f'{mean:.6f}'}, "
        f"std={'nan' if math.isnan(std) else f'{std:.6f}'}"
    )

if missing_triplets:
    summary_lines.append("\nMissing job triples (algo, task_type, task):")
    summary_lines.extend([f"{a}\t{tt}\t{t}" for a, tt, t in missing_triplets])

summary_file = os.path.join(log_path, "eval_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"Aggregation done. Progress: {progress_file}")
PY

  echo -e "${C_BOLD}${C_GREEN}[DONE] All jobs finished.${C_RESET}"
  echo -e "${C_GREEN}[DONE] Raw progress: ${PROGRESS_FILE}${C_RESET}"
  echo -e "${C_GREEN}[DONE] Global summary: ${LOG_PATH}/eval_summary.txt${C_RESET}"
  echo -e "${C_GREEN}[DONE] Per algorithm summary: ${LOG_PATH}/{Algorithm}/demo_clean.txt or demo_randomized.txt${C_RESET}"
