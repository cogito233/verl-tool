# ray stop
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head --dashboard-host=0.0.0.0 
source .venv-server/bin/activate
export WANDB_ENTITY=zhihenglyu-cs
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1

set -x
# dataset_name=r2e_swe_debug
dataset_name=r2e_sync_extra
# dataset_name=r2e_swe_extra_debug
train_data=data/$dataset_name/train.parquet
val_data=data/$dataset_name/train.parquet
model_name=R2EGym-32B-Agent
model_path=/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-32B-Agent
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=8
n_nodes=1
enable_agent=True # enable agent for tool use

# n=8
# batch_size=32
# 可能挂了

# n=8
# batch_size=16

n=4
batch_size=8   

ppo_mini_batch_size=8
max_prompt_length=2048
max_response_length=20480 
max_obs_length=8192
temperature=0.5
strategy="fsdp" # remove _agent for normal verl behavior
valid_actions="[]" 
token of
# each action, which are </answer> and </python> respectively

# === begin, added by Zhiheng ===
rollout_mode='sync'
max_action_length=2048
rolling_with_prompt=False
call_tool_first=True
truncate_obs_side=left # This is weird but required in the current code
truncate_response_side=left
min_action_num=5
mirco_batch_size=1
mirco_batch_size_non_train=1
max_start_length=2048 # System prompt is always length 800+, not the bottleneck
use_dynamic_bsz=True # faster
enable_mtrl=True
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
# === end, added by Zhiheng ===

lr_multiple=1
critic_lr=5e-7
actor_lr=1e-6

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="${model_pretty_name}-${dataset_name}-baseline-0623-ST-aync"
export VERL_RUN_ID=$run_name
export VLLM_ATTENTION_BACKEND=XFORMERS

host=localhost
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
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
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
    +actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=$max_start_length \
    +actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_action_length=$max_action_length \
    +actor_rollout_ref.agent.rolling_with_prompt=$rolling_with_prompt \
    +actor_rollout_ref.agent.call_tool_first=call_tool_first \
    +actor_rollout_ref.agent.min_action_num=$min_action_num \
    +actor_rollout_ref.agent.truncate_response_side=$truncate_response_side \
    +actor_rollout_ref.agent.truncate_obs_side=$truncate_obs_side \
    +actor_rollout_ref.agent.max_turns=40 \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.valid_actions=$valid_actions \
    +actor_rollout_ref.agent.no_action_as_stop=False \
    +actor_rollout_ref.actor.enable_agent=$enable_agent \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$mirco_batch_size_non_train \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
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
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1


pkill -P -9 $server_pid
kill -9 $kill $server_pid

# loss from 1e-5 to 5e-7;
