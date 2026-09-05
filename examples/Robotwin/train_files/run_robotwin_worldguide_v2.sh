

export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# Debug and logging
#export NCCL_DEBUG=WARN
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_SHOW_CPP_STACKTRACES=1
#export CUDA_LAUNCH_BLOCKING=1
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFTLeWorldV2Robo
freeze_module_list='qwenfast_model.qwen_vl_interface.model.visual'
base_vlm=/H20_vepfs/liulin/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=/H20_vepfs/liulin/starVLA/examples/Robotwin/train_files/starvla_worldguide_robotwin_V2.yaml
robotwin_data_root=/H20_vepfs/liulin/starVLA/datasets/Robotwin
data_mix=robotwin_lerobot_mix_50
run_root_dir=./results_custom/Checkpoints
run_id=1229_robotwin_qwen3oftworldv2
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

LOG_FILE=${output_dir}/train_$(date +%Y%m%d_%H%M%S).log

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${robotwin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 140000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000 \
  --trainer.is_resume True \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  2>&1 | tee ${LOG_FILE}

echo "Exit code: $?" >> ${LOG_FILE}
