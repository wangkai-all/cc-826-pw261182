#!/usr/bin/env bash
# 启动罕见事件分析 Web 应用（仅依赖 Python 3 标准库）
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m backend.cli serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"
