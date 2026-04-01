#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行所有 download_*.py 脚本
for script in download_*.py; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        python "$script"
    fi
done
