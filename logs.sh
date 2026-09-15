#!/bin/bash
# logs.sh — 实时查看日志
cd "$(dirname "$0")" || exit 1

case "${1:-all}" in
    front)   tail -f logs/front.log ;;
    bottom)  tail -f logs/bottom.log ;;
    web)     tail -f logs/web_server.log ;;
    legacy)  tail -f logs/legacy_bridge.log ;;
    all)     tail -f logs/*.log ;;
    *)       echo "用法: $0 [front|bottom|web|legacy|all]"; exit 1 ;;
esac