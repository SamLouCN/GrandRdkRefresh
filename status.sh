#!/bin/bash
# ============================================================
# status.sh — 查看 GrandRdkRefresh 运行状态
# ============================================================
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
CONFIG_DIR="$ROOT/config"

echo "========== 进程 =========="
for name in front.py bottom.py web_server.py legacy_bridge.py; do
    pids=$(pgrep -f "src/$name" 2>/dev/null | tr '\n' ' ')
    if [ -n "$pids" ]; then
        printf "  %-20s [ON ] pid=%s\n" "$name" "$pids"
    else
        printf "  %-20s [OFF]\n" "$name"
    fi
done

echo ""
echo "========== 共享内存 =========="
ls -lh /dev/shm/momo_* 2>/dev/null || echo "  (无)"

echo ""
echo "========== Web 服务 =========="
curl -s -o /dev/null -w "  web_server :5000   -> %{http_code}\n" \
    http://127.0.0.1:5000/api/status 2>/dev/null || echo "  web_server :5000   -> 无响应"

echo ""
echo "========== Nginx =========="
curl -s -o /dev/null -w "  nginx      :80     -> %{http_code}\n" \
    http://127.0.0.1/ 2>/dev/null || echo "  nginx      :80     -> 无响应"

echo ""
echo "========== 日志尾部 =========="
for f in "$ROOT"/logs/*.log; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    tail -n 3 "$f" 2>/dev/null || echo "  (空)"
    echo ""
done