#!/bin/bash
# ============================================================
# run.sh — GrandRdkRefresh 统一启动脚本
#
# 启动:
#   front.py       前视检测 (CPU 0-2)  -> 写共享内存
#   bottom.py      下视检测 (CPU 3-5)  -> 写共享内存
#   web_server.py  FastAPI :5000       -> 读共享内存
#   legacy_bridge.py (可选)             -> 旧协议兼容 :9000/:9001/:8081
#   Nginx          :80                  -> 静态 + 反代
#
# 用法:
#   ./run.sh                     # 真实相机 + 全部启动
#   ./run.sh --virtual           # 虚拟相机 (无硬件联调)
#   ./run.sh --frames 100        # 每个检测进程跑 100 帧
#   ./run.sh --no-web            # 只跑检测, 不起 Web/Nginx
#   ./run.sh --no-legacy         # 不启动 legacy 兼容层
#   ./run.sh --setup-only        # 只做初始化 (依赖检查 + Nginx 配置)
#   ./run.sh --no-nginx          # 不 reload Nginx
# ============================================================
set -u

# ---------- 项目路径 ----------
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1

CONFIG_DIR="$ROOT/config"
SRC_DIR="$ROOT/src"
WEB_DIR="$ROOT/web"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------- 环境变量 ----------
export PYTHONPATH="$CONFIG_DIR:$SRC_DIR:$SRC_DIR/utils"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

# ---------- 默认参数 ----------
VIRTUAL=0
FRAMES=0
NO_WEB=0
NO_LEGACY=0
NO_NGINX=0
SETUP_ONLY=0
DETECT_ARGS=()

# ---------- 解析参数 ----------
while [ $# -gt 0 ]; do
    case "$1" in
        --virtual)   VIRTUAL=1; DETECT_ARGS+=("$1"); shift ;;
        --frames)    FRAMES="$2"; DETECT_ARGS+=("$1" "$2"); shift 2 ;;
        --frames=*)  FRAMES="${1#*=}"; DETECT_ARGS+=("$1"); shift ;;
        --no-web)    NO_WEB=1; shift ;;
        --no-legacy) NO_LEGACY=1; shift ;;
        --no-nginx)  NO_NGINX=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            # 其他参数透传给 front/bottom
            DETECT_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------- 工具函数 ----------
info() { echo "[run.sh] $*"; }
warn() { echo "[run.sh] [!] $*"; }
err()  { echo "[run.sh] [X] $*" >&2; }

# ---------- 读取 main_config 开关 ----------
read_cfg() {
    python3 -c "import sys; sys.path.insert(0,'$CONFIG_DIR'); import main_config as m; print(getattr(m, '$1', ''))"
}

# ============================================================
# 0. 依赖检查
# ============================================================
info "项目根: $ROOT"

# 检查关键文件
for f in "$CONFIG_DIR/main_config.py" "$SRC_DIR/front.py" "$SRC_DIR/bottom.py"; do
    if [ ! -f "$f" ]; then
        err "缺少文件: $f"
        exit 1
    fi
done
info "关键文件检查通过"

# Python 依赖
if [ "$NO_WEB" = "0" ]; then
    if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
        warn "缺少 fastapi/uvicorn, 尝试安装..."
        pip3 install -q fastapi uvicorn websockets 2>/dev/null || {
            err "安装失败, 请手动: pip3 install fastapi uvicorn websockets"
            exit 1
        }
    fi
    info "Web 依赖检查通过"
fi

# ============================================================
# 1. 配置 Nginx (首次)
# ============================================================
if [ "$NO_WEB" = "0" ] && [ "$NO_NGINX" = "0" ]; then
    if [ ! -L /etc/nginx/sites-enabled/vp ]; then
        info "首次运行, 配置 Nginx"
        if [ -x "$ROOT/nginx_setup.sh" ]; then
            "$ROOT/nginx_setup.sh" || warn "nginx_setup.sh 失败, 继续"
        else
            warn "未找到 nginx_setup.sh, 跳过 Nginx 配置"
        fi
    fi
fi

if [ "$SETUP_ONLY" = "1" ]; then
    info "--setup-only 完成"
    exit 0
fi

# ============================================================
# 2. 清理旧进程
# ============================================================
info "清理旧进程..."
pkill -f "src/front.py"        2>/dev/null
pkill -f "src/bottom.py"       2>/dev/null
pkill -f "src/web_server.py"   2>/dev/null
pkill -f "src/legacy_bridge.py" 2>/dev/null
sleep 0.5

# 清理旧的共享内存（避免读到脏数据）
rm -f /dev/shm/momo_*.bin /dev/shm/momo_*.json 2>/dev/null

# ============================================================
# 3. 启动检测进程
# ============================================================
info "启动 front.py (CPU 0-2)"
taskset -c 0-2 python3 "$SRC_DIR/front.py" "${DETECT_ARGS[@]}" \
    > "$LOG_DIR/front.log" 2>&1 &
PID_FRONT=$!

info "启动 bottom.py (CPU 3-5)"
taskset -c 3-5 python3 "$SRC_DIR/bottom.py" "${DETECT_ARGS[@]}" \
    > "$LOG_DIR/bottom.log" 2>&1 &
PID_BOTTOM=$!

sleep 2

# 检测进程是否活着
if ! kill -0 $PID_FRONT 2>/dev/null; then
    err "front.py 启动失败, 查看 $LOG_DIR/front.log"
    tail -n 20 "$LOG_DIR/front.log"
    exit 1
fi
if ! kill -0 $PID_BOTTOM 2>/dev/null; then
    err "bottom.py 启动失败, 查看 $LOG_DIR/bottom.log"
    tail -n 20 "$LOG_DIR/bottom.log"
    exit 1
fi
info "检测进程已启动: front=$PID_FRONT bottom=$PID_BOTTOM"

# ============================================================
# 4. 启动 Web 服务
# ============================================================
PID_WEB=""
PID_LEGACY=""

if [ "$NO_WEB" = "0" ]; then
    EN_NEW=$(read_cfg ENABLE_WEB_NEW)
    EN_LEGACY=$(read_cfg ENABLE_WEB_LEGACY)

    if [ "$EN_NEW" = "True" ]; then
        info "启动 web_server.py (FastAPI :5000)"
        nice -n 10 python3 "$SRC_DIR/web_server.py" \
            > "$LOG_DIR/web_server.log" 2>&1 &
        PID_WEB=$!
        sleep 1
        if ! kill -0 $PID_WEB 2>/dev/null; then
            err "web_server.py 启动失败, 查看 $LOG_DIR/web_server.log"
            tail -n 20 "$LOG_DIR/web_server.log"
        else
            info "web_server 已启动: pid=$PID_WEB"
        fi
    else
        info "ENABLE_WEB_NEW=False, 跳过 web_server"
    fi

    if [ "$NO_LEGACY" = "0" ] && [ "$EN_LEGACY" = "True" ]; then
        if [ -f "$SRC_DIR/legacy_bridge.py" ]; then
            info "启动 legacy_bridge.py (:9000/:9001/:8081)"
            nice -n 10 python3 "$SRC_DIR/legacy_bridge.py" \
                > "$LOG_DIR/legacy_bridge.log" 2>&1 &
            PID_LEGACY=$!
        else
            warn "未找到 legacy_bridge.py, 跳过"
        fi
    fi

    # ---------- Nginx reload ----------
    if [ "$NO_NGINX" = "0" ]; then
        info "reload Nginx"
        sudo nginx -t 2>/dev/null && \
            (sudo systemctl reload nginx 2>/dev/null || sudo nginx -s reload 2>/dev/null) \
            || warn "Nginx reload 失败"
    fi
fi

# ============================================================
# 5. 打印访问信息
# ============================================================
BOARD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$BOARD_IP" ] && BOARD_IP="<板卡IP>"

echo ""
info "============================================"
info " 启动完成"
info " 前视 pid : $PID_FRONT"
info " 下视 pid : $PID_BOTTOM"
[ -n "$PID_WEB" ]    && info " Web  pid : $PID_WEB"
[ -n "$PID_LEGACY" ] && info " Leg  pid : $PID_LEGACY"
info " 日志目录 : $LOG_DIR"
if [ "$NO_WEB" = "0" ]; then
    info " 浏览器   : http://$BOARD_IP/"
    info " MJPEG    : http://$BOARD_IP/cam1  http://$BOARD_IP/cam2"
    info " API      : http://$BOARD_IP/api/status"
    info " WS       : ws://$BOARD_IP/ws/status"
fi
info "============================================"
echo ""

# ============================================================
# 6. 等待 / 退出
# ============================================================
cleanup() {
    echo ""
    info "收到停止信号, 清理进程..."
    kill $PID_FRONT $PID_BOTTOM $PID_WEB $PID_LEGACY 2>/dev/null
    wait 2>/dev/null
    info "全部已停止"
    exit 0
}
trap cleanup INT TERM

# 等待任一检测进程退出（有限帧模式会自然退出）
wait $PID_FRONT $PID_BOTTOM 2>/dev/null
info "检测进程已退出"
kill $PID_WEB $PID_LEGACY 2>/dev/null
wait 2>/dev/null
info "run.sh 结束"