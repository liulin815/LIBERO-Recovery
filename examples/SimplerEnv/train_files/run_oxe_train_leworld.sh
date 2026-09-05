export NCCL_SOCKET_IFNAME=eth0
#export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TLeWorld
# QwenGR00TLeWorld has no `qwenfast_model` wrapper: the VLM lives directly on
# self.qwen_vl_interface (same freeze target as QwenOFTLeWorldSim's
# 'qwenfast_model.qwen_vl_interface.model.visual').
freeze_module_list='qwen_vl_interface.model.visual'
base_vlm=/H20_vepfs/liulin/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=/H20_vepfs/liulin/starVLA/examples/SimplerEnv/train_files/starvla_cotrain_oxe_worldguide_leworld.yaml
oxe_data_root=/H20_vepfs/liulin/starVLA/datasets/OXE_LEROBOT
data_mix=bridge_rt_1
run_root_dir=./results_simpler_env/Checkpoints
run_id=0819_${data_mix}_${Framework_name}
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
  starVLA/training/train_starvla_simpler_env.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  # --wandb_project starVLA_simplerEnv \
  # --wandb_entity jinhuiye \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
