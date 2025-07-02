#!/usr/bin/env bash
set -euo pipefail

log_server=job_logs/bash_32B_8node_server.log
log_train=job_logs/bash_32B_8node_train.log
job_server=examples/train/r2e_swe/r2e_server_mnode.sh
job_train=examples/train/r2e_r2e/train_r2eagent_32B_bs256_32K_8node_swe_val.sh


# 📝 若要下载 punkt_tab 就把下面三行取消注释
# .venv/bin/python - <<'PY'
# import nltk; nltk.download("punkt_tab")
# PY

cd /root/code/rl_r2e

ray status
sleep 120
ray status

# --------- 工具函数 ----------
# 前台跑 + 日志
run_fg () {
  local logf=$1; shift
  echo "[`date`] ▶️ $*" | tee -a "$logf"
  "$@" 2>&1 | tee -a "$logf"
}

# 后台跑 + 日志（不返回值，用 $! 拿 PID）
run_bg () {
  local logf=$1; shift
  echo "[`date`] ▶️ $*" | tee -a "$logf"
  (
    exec "$@" 2>&1 | tee -a "$logf"
  ) &                                # 真后台
}
# -----------------------------

# 1️⃣ 启动 server（后台）
run_bg $log_server bash $job_server
server_pid=$!                       # ← 这里直接拿后台进程 PID
echo "[`date`] 🏷 server PID=$server_pid"

# 2️⃣ 等 60 秒热身
sleep 60

# 3️⃣ 启动训练（前台）
run_fg $log_train bash $job_train

# 4️⃣ 训练结束后杀掉 server
echo "[`date`] 🛑 training done, killing server PID=$server_pid"
kill -9 "$server_pid" 2>/dev/null || true
pkill -f "$job_server" 2>/dev/null || true
sleep 5

echo "[`date`] 🎉 all tasks finished，收工！"
