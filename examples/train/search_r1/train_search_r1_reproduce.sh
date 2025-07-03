#!/bin/bash

# Search-R1 style training with verl-tool
# This script demonstrates how to train an LLM to use search capabilities

set -x

# Model and data configuration
# or download at: https://huggingface.co/Qwen/Qwen2.5-3B/tree/main
model_name="Qwen/Qwen2.5-3B"
train_data="./data/search_r1/training_data/searchR1_processed_direct/train.parquet"
val_data="./data/search_r1/training_data/searchR1_processed_direct/test.parquet"

# Search-R1 specific action tokens
action_stop_tokens="</search>,</answer>"
retriever_url="http://127.0.0.1:8000/retrieve"
retriever_topk=3
total_epochs=15
total_training_steps=1005

# Training hyperparameters
rl_alg=grpo # gae(ppo) or grpo
n_gpus_per_node=8
n_nodes=1
n=16
batch_size=512
ppo_mini_batch_size=256
max_prompt_length=4096
max_response_length=2560    # 500
data_max_response_length=3000
max_obs_length=1024 # 512
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp" # fsdp2
max_turns=2
kl_loss_coef=0.001
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=search_r1_qa_em
ppo_micro_batch_size_per_gpu=8  # 8 * 8=64, match train_grpo.sh in Search-R1
log_prob_micro_batch_size_per_gpu=16    # 8 * 16=128, match train_grpo.sh in Search-R1
tensor_model_parallel_size=1
gpu_memory_utilization=0.6 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=True # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training
max_action_length=2048
# Generate run name
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="search_r1"
run_name="new_data_sglang_retriever_search_r1_qa_em-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

# # Launch retrieval server (same as Search-R1)
# echo "Starting retrieval server..."
# cd /map-vepfs/yi/Search-R1
# export HF_ENDPOINT=https://hf-mirror.com
# 
# conda activate retriever
# bash /map-vepfs/yi/Search-R1/retrieval_launch.sh &
# retriever_pid=$!
# cd - # Return to original directory
# conda activate verl-tool-env-yi # activate verl-tool environment
# sleep 10  # Wait for retrieval server to start

# Set environment variables for search tool configuration
export RETRIEVER_URL=$retriever_url
export RETRIEVER_TOPK=$retriever_topk

# Create action stop tokens file
mkdir -p $(pwd)/tmp
action_stop_tokens_file="$(pwd)/tmp/search_r1_action_tokens.txt"
echo -n "$action_stop_tokens" > $action_stop_tokens_file
echo "Action stop tokens file: $action_stop_tokens_file"

echo "Retrieval server started with PID: $retriever_pid"

# Launch tool server with search capabilities
host=$(hostname -I | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
# Start tool server with both search_retrieval and finish tools
python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "search_retrieval,finish" \
    --workers_per_tool 32 &
tool_server_pid=$!

echo "Tool server (pid=$tool_server_pid) started at $tool_server_url"

# TODO: fix the error of cannot invoke:
# actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model'] \

# Run training
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=256 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$data_max_response_length \
    +data.max_start_length=2048 \
    +data.max_obs_length=500 \
    +data.shuffle_train_dataloader=True \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    +actor_rollout_ref.actor.enable_agent=$enable_agent \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_turns=$max_turns \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    +actor_rollout_ref.agent.mask_observations=$mask_observations \
    +actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    +actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    +actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=20 \
    trainer.test_freq=40 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \

# Cleanup
echo "Training completed. Cleaning up..."
pkill -P -9 $tool_server_pid
kill -9 $tool_server_pid
echo "Cleanup completed." 