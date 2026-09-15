#!/bin/bash
# ============================================================
# stop.sh — 停止 GrandRdkRefresh 所有进程
# ============================================================
set -u
cd "$(dirname "$0")" || exit 1

echo "[stop.sh] 停止检测与 Web 进程..."
pkill -f "src/front.py"         2>/dev/null
pkill -f "src/bottom.py"        2>/dev/null
pkill -f "src/web_server.py"    2>/dev/null
pkill -f "src/legacy_bridge.py" 2>/dev/null

sleep 0.5
echo "[stop.sh] 清理共享内存..."
rm -f /dev/shm/momo_*.bin /dev/shm/momo_*.json 2>/dev/null

echo "[stop.sh] 完成"