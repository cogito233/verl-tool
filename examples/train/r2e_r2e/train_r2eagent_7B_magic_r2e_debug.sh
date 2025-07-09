# ray stop
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head --dashboard-host=0.0.0.0 
source .venv-server/bin/activate
export WANDB_ENTITY=zhihenglyu-cs
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

set -x
# dataset_name=r2e_swe_debug
dataset_name=r2e_swe_verified_user
# dataset_name=r2e_swe_extra_debug
# train_data=/root/code/rl_r2e/data/$dataset_name/test.parquet
# val_data=/root/code/rl_r2e/data/$dataset_name/test.parquet
train_data=/root/code/rl_r2e/data/r2e_lite_user/test.parquet
val_data=/root/code/rl_r2e/data/r2e_lite_user/test.parquet
model_name=R2EGym-7B-Agent
model_path=/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=4
n_nodes=1
enable_agent=True # enable agent for tool use

# n=8
# batch_size=32
n=4
batch_size=4

ppo_mini_batch_size=4
max_prompt_length=10240
max_response_length=22527 
max_model_length=32768
max_obs_length=4096
temperature=1.0
strategy="fsdp" # remove _agent for normal verl behavior
valid_actions="[]" 
# token of each action, which are </answer> and </python> respectively

# === begin, added by Zhiheng ===
rollout_mode='async'
max_action_length=1536
rolling_with_prompt=False
call_tool_first=False
truncate_obs_side=left # This is weird but required in the current code
truncate_response_side=left
min_action_num=5
mirco_batch_size=1
mirco_batch_size_non_train=1
max_start_length=2047 # System prompt is always length 800+, not the bottleneck
use_dynamic_bsz=True # faster
enable_mtrl=True
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
do_offload=True
fsdp_size=-1
# === end, added by Zhiheng ===

critic_lr=5e-7
actor_lr=1e-6

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${model_pretty_name}-${dataset_name}-debug"
export VERL_RUN_ID=$run_name
export VLLM_ATTENTION_BACKEND=XFORMERS

# host=localhost
host=$(hostname -I | awk '{print $1}')
# # port=$(shuf -i 30000-31000 -n 1)
port=30815
tool_server_url=http://$host:$port/get_observation
# python -m verl_tool.servers.serve --host $host --port $port --tool_type "text_browser" &
# server_pid=$!
# echo "Server (pid=$server_pid) started at $tool_server_url"

# actor_rollout_ref.rollout.enforce_eager=False \
# actor_rollout_ref.rollout.free_cache_engine=False \

# export VLLM_USE_V1=1
# actor_rollout_ref.agent.max_turns is for debug only

# ray job submit --address="http://127.0.0.1:8265" \
#     --runtime-env=verl_tool/trainer/runtime_env.yaml \
#     -- \
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    reward_model.reward_manager=r2eswe \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$mirco_batch_size \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_model_length=$max_model_length \
    actor_rollout_ref.agent.max_start_length=$max_start_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.rolling_with_prompt=$rolling_with_prompt \
    actor_rollout_ref.agent.call_tool_first=$call_tool_first \
    +actor_rollout_ref.agent.min_action_num=$min_action_num \
    actor_rollout_ref.agent.truncate_response_side=$truncate_response_side \
    actor_rollout_ref.agent.truncate_obs_side=$truncate_obs_side \
    actor_rollout_ref.agent.mask_overlong_loss=True \
    actor_rollout_ref.agent.max_turns=40 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.valid_actions=$valid_actions \
    +actor_rollout_ref.agent.no_action_as_stop=False \
    +actor_rollout_ref.actor.enable_agent=$enable_agent \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$mirco_batch_size_non_train \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=$max_model_length \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$mirco_batch_size_non_train \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=$critic_lr \
    critic.strategy=$strategy \
    critic.model.path=$model_path \
    critic.ppo_micro_batch_size_per_gpu=$mirco_batch_size \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='r2e_swe' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$(pwd)/checkpoints/r2eswe/${run_name} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=5


# pkill -P -9 $server_pid
# kill -9 $kill $server_pid

# loss from 1e-5 to 5e-7;
