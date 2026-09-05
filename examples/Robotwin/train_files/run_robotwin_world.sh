export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=SingleWMRobo
freeze_module_list=''
config_yaml=./examples/Robotwin/train_files/starvla_robotwin_world.yaml
run_root_dir=./results/Checkpoints_robo_WM
data_mix=robotwin_lerobot_mix_success_failure
run_id=0129_${data_mix}_qwen3OFT_WM
robotwin_data_root=/H20_vepfs/liulin/starVLA/datasets/Robotwin
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_leworldmodel.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.data_root_dir ${robotwin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 70000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.resume True \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
