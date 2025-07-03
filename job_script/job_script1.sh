#!/usr/bin/env bash
set -euo pipefail

# 📝 若要下载 punkt_tab 就把下面三行取消注释
# .venv/bin/python - <<'PY'
# import nltk; nltk.download("punkt_tab")
# PY

cd /root/code/rl_r2e

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
run_bg bash_async1.log bash examples/train/r2e_swe/r2e_server_async.sh
server_pid=$!                       # ← 这里直接拿后台进程 PID
echo "[`date`] 🏷 server PID=$server_pid"

# 2️⃣ 等 60 秒热身
sleep 60

# 3️⃣ 启动训练（前台）
run_fg bash_async2.log bash examples/train/r2e_swe/train_r2eagent_7B_bs32_20K_async.sh

# 4️⃣ 训练结束后杀掉 server
echo "[`date`] 🛑 training done, killing server PID=$server_pid"
kill -TERM "$server_pid" 2>/dev/null || true
sleep 5
kill -KILL "$server_pid" 2>/dev/null || true

echo "[`date`] 🎉 all tasks finished，收工！"
