

export NCCL_SOCKET_IFNAME=eth0
#export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list='qwen_vl_interface.model.visual'
base_vlm=/H20_vepfs/liulin/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=/H20_vepfs/liulin/starVLA/examples/LIBERO/train_files/starvla_overfit_10ep.yaml
libero_data_root=/H20_vepfs/liulin/starVLA/datasets/LEROBOT_LIBERO_DATA
data_mix=libero_90
run_root_dir=./results_custom/Checkpoints
run_id=qwen3oft_baseline_90
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.num_warmup_steps 3000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  #--trainer.is_resume True \
