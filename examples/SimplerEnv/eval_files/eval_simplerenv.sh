#!/usr/bin/env bash
set -euo pipefail

###################################
# User-configurable parameters
###################################
CONDA_SH="/H20_vepfs/liulin/miniconda3/etc/profile.d/conda.sh"
SERVER_ENV="starvla"
CLIENT_ENV="simpler_env"
source "${CONDA_SH}"

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SimplerEnv_PATH="${SimplerEnv_PATH:-/H20_vepfs/liulin/SimplerEnv}"
SIMPLER_ENV_LIB_DIR="${SIMPLER_ENV_LIB_DIR:-}"

GPUS_CSV="${GPUS_CSV:-0}"
BASE_PORT="${BASE_PORT:-6678}"
your_ckpt="${your_ckpt:-./results/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt}"
USE_BF16="${USE_BF16:-1}"
SERVER_WARMUP_SECONDS="${SERVER_WARMUP_SECONDS:-60}"
TSET_NUM="${TSET_NUM:-1}"

# Dataset saving options
SAVE_DATASET="${SAVE_DATASET:-0}"           # Set to 1 to save eval data in LeRobot format
DATASET_OUT_PATH="${DATASET_OUT_PATH:-}"    # Output dir (default: <ckpt_dir>/eval_dataset_lerobot)
DATASET_IMAGE_SIZE="${DATASET_IMAGE_SIZE:-256 256}"  # H W for saved video frames

###################################
# Color helpers
###################################
C_RESET='\033[0m'
C_RED='\033[31m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_MAGENTA='\033[35m'
C_CYAN='\033[36m'
C_BOLD='\033[1m'

###################################
# Parse GPUs
###################################
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
NUM_GPUS=${#GPUS[@]}
echo -e "${C_BOLD}${C_CYAN}========== SimplerEnv Multi-GPU Evaluation ==========${C_RESET}"
echo -e "${C_GREEN}[INFO] GPUs: ${GPUS_CSV} (${NUM_GPUS} GPUs)${C_RESET}"
echo -e "${C_GREEN}[INFO] Checkpoint: ${your_ckpt}${C_RESET}"

###################################
# Setup paths
###################################
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
if [[ -n "${SIMPLER_ENV_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SIMPLER_ENV_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_server_dir}" "${output_eval_dir}"

###################################
# Cleanup on exit
###################################
SERVER_PIDS=()
cleanup() {
  set +e
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo -e "${C_YELLOW}[CLEANUP] Killing policy server (PID=${pid})${C_RESET}"
      kill "${pid}" >/dev/null 2>&1
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

###################################
# Step 1: Start policy servers (one per GPU)
###################################
declare -a PORTS=()

for ((g=0; g<NUM_GPUS; g++)); do
  gpu="${GPUS[$g]}"
  port=$((BASE_PORT + g))
  PORTS+=("${port}")
  server_log="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

  echo -e "${C_GREEN}[SERVER] Starting server on GPU=${gpu}, port=${port}${C_RESET}"

  SERVER_CMD="python deployment/model_server/server_policy.py --ckpt_path ${your_ckpt} --port ${port}"
  if [[ "${USE_BF16}" == "1" ]]; then
    SERVER_CMD+=" --use_bf16"
  fi

  (
    set +u
    conda activate "${SERVER_ENV}"
    set -u
    CUDA_VISIBLE_DEVICES="${gpu}" ${SERVER_CMD} 2>&1 | tee "${server_log}"
  ) &
  SERVER_PIDS+=("$!")
done

echo -e "${C_MAGENTA}[INFO] ${NUM_GPUS} server(s) started, waiting ${SERVER_WARMUP_SECONDS}s for warmup...${C_RESET}"
sleep "${SERVER_WARMUP_SECONDS}"

for ((g=0; g<NUM_GPUS; g++)); do
  if ! kill -0 "${SERVER_PIDS[$g]}" >/dev/null 2>&1; then
    echo -e "${C_RED}[ERROR] Server on GPU=${GPUS[$g]} port=${PORTS[$g]} died during warmup.${C_RESET}"
    exit 1
  fi
done
echo -e "${C_GREEN}[SERVER] All ${NUM_GPUS} server(s) are running.${C_RESET}"

###################################
# Step 2: Build task list
###################################
declare -a ALL_TASKS=()

# V1 scenes
scene_name_v1=bridge_table_1_v1
robot_v1=widowx
rgb_overlay_v1=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x_v1=0.147
robot_init_y_v1=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for env in "${ENV_NAMES[@]:-}"; do
  [[ -n "${env}" ]] || continue
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    ALL_TASKS+=("v1|${env}|${run_idx}")
  done
done

# V2 scenes
scene_name_v2=bridge_table_1_v2
robot_v2=widowx_sink_camera_setup
rgb_overlay_v2=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x_v2=0.127
robot_init_y_v2=0.06

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

for env in "${ENV_NAMES_V2[@]:-}"; do
  [[ -n "${env}" ]] || continue
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    ALL_TASKS+=("v2|${env}|${run_idx}")
  done
done

NUM_TASKS=${#ALL_TASKS[@]}
echo -e "${C_CYAN}[EVAL] Total tasks: ${NUM_TASKS}, distributing across ${NUM_GPUS} GPU(s)${C_RESET}"

###################################
# Step 3: Run evaluations (round-robin across GPUs)
###################################
for ((t=0; t<NUM_TASKS; t++)); do
  task="${ALL_TASKS[$t]}"
  gpu_idx=$((t % NUM_GPUS))
  port="${PORTS[$gpu_idx]}"
  gpu="${GPUS[$gpu_idx]}"

  IFS='|' read -r version env run_idx <<< "${task}"

  if [[ "${version}" == "v1" ]]; then
    scene_name="${scene_name_v1}"
    robot="${robot_v1}"
    rgb_overlay_path="${rgb_overlay_v1}"
    robot_init_x="${robot_init_x_v1}"
    robot_init_y="${robot_init_y_v1}"
  else
    scene_name="${scene_name_v2}"
    robot="${robot_v2}"
    rgb_overlay_path="${rgb_overlay_v2}"
    robot_init_x="${robot_init_x_v2}"
    robot_init_y="${robot_init_y_v2}"
  fi

  task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
  echo -e "${C_CYAN}[EVAL] Task [${env}] run#${run_idx} → GPU=${gpu} port=${port}${C_RESET}"

  # Build save-dataset flags
  SAVE_DATASET_FLAGS=""
  if [[ "${SAVE_DATASET}" == "1" ]]; then
    SAVE_DATASET_FLAGS="--save-dataset"
    if [[ -n "${DATASET_OUT_PATH}" ]]; then
      SAVE_DATASET_FLAGS+=" --dataset-out-path ${DATASET_OUT_PATH}"
    else
      SAVE_DATASET_FLAGS+=" --dataset-out-path ${ckpt_dir}/eval_dataset_lerobot"
    fi
    SAVE_DATASET_FLAGS+=" --dataset-image-size ${DATASET_IMAGE_SIZE}"
  fi

  conda run --no-capture-output -n "${CLIENT_ENV}" \
    env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    xvfb-run -a python examples/SimplerEnv/eval_files/start_simpler_env_with_save.py \
      --ckpt-path "${your_ckpt}" \
      --port "${port}" \
      --robot "${robot}" \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name "${scene_name}" \
      --rgb-overlay-path "${rgb_overlay_path}" \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 500 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      ${SAVE_DATASET_FLAGS} \
      > "${task_log}" 2>&1 &

  sleep 6
done

wait

echo -e "${C_BOLD}${C_GREEN}[DONE] All SimplerEnv evaluations finished.${C_RESET}"
echo -e "${C_GREEN}[DONE] Eval logs: ${output_eval_dir}/${C_RESET}"
