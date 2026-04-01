#!/bin/bash


export CUDA_VISIBLE_DEVICES=1

export OLLAMA_HOST=127.0.0.1:6006

echo "正在检查端口 6006..."

PID=$(lsof -t -i:6006)
if [ -n "$PID" ]; then
    echo "端口 6006 被进程 $PID 占用，正在清理..."
    kill -9 $PID
    sleep 2 
fi

echo "正在启动 Ollama 服务 (端口: 6006, 显卡: CUDA 1)..."
nohup ollama serve > ollama_server.log 2>&1 &

sleep 5

echo "正在运行模型 qwen3.5:27b..."
ollama run qwen3.5:27b