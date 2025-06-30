#!/bin/bash
source .venv-server/bin/activate

export VLLM_USE_V1=1
export NCCL_DEBUG=INFO

# 停止现有的 ray 和服务器
ray stop
pkill -f "verl_tool.servers.serve"

# 启动 ray 头节点
ray start --head --dashboard-host=0.0.0.0

# 工具服务器配置
host=$(hostname -I | awk '{print $1}')  # 获取本机IP，供其他节点访问
port=30815
tool_server_url=http://$host:$port/get_observation

echo "Starting tool server at $tool_server_url"

# 启动工具服务器（带重试机制）
python -m verl_tool.servers.serve --host $host --port $port --tool_type sandbox_r2e --workers_per_tool 64 &
server_pid=$!
sleep 5

# 检查服务器是否启动成功
if kill -0 $server_pid 2>/dev/null; then
    echo "Tool server (pid=$server_pid) started successfully at $tool_server_url"
    echo "Server PID: $server_pid" > /tmp/tool_server.pid
    echo "Server URL: $tool_server_url" > /tmp/tool_server.url
else
    echo "Failed to start tool server"
    exit 1
fi

# 保持脚本运行，等待手动终止
echo "Server is running. Press Ctrl+C to stop..."
trap "echo 'Stopping server...'; kill -9 $server_pid; ray stop; exit" INT

# 等待信号
while kill -0 $server_pid 2>/dev/null; do
    sleep 10
done

echo "Server process ended unexpectedly"
ray stop