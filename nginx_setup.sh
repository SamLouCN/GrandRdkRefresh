#!/bin/bash
# ============================================================
# nginx_setup.sh — 自动配置 Nginx 站点 (项目根目录执行)
#
# 作用:
#   1) 检查 Nginx 是否安装
#   2) 把 config/nginx/vp.conf 软链到 /etc/nginx/sites-enabled/vp
#   3) 删除默认站点 (sites-enabled/default)，避免抢 80 端口
#   4) 语法检查 + reload/启动 Nginx
#
# 用法:
#   ./nginx_setup.sh              # 安装并 reload
#   ./nginx_setup.sh --uninstall  # 卸载软链并 reload
#   ./nginx_setup.sh --dry-run    # 只打印将要做什么, 不实际执行
# ============================================================
set -u

# ---------- 参数 ----------
DRY_RUN=0
UNINSTALL=0
for a in "$@"; do
    case "$a" in
        --dry-run)   DRY_RUN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *) echo "[!] 未知参数: $a"; exit 2 ;;
    esac
done

# ---------- 常量 ----------
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC_CONF="$ROOT/config/nginx/vp.conf"
SITE_NAME="vp"

NGINX_SITES_ENABLED="/etc/nginx/sites-enabled"
NGINX_SITES_AVAILABLE="/etc/nginx/sites-available"
DEST_LINK="$NGINX_SITES_ENABLED/$SITE_NAME"

# ---------- 工具函数 ----------
info() { echo "[*] $*"; }
warn() { echo "[!] $*"; }
err()  { echo "[X] $*" >&2; }

run() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [dry-run] $*"
    else
        eval "$@"
    fi
}

# ---------- 0. 检查 Nginx 是否存在 ----------
if ! command -v nginx >/dev/null 2>&1; then
    err "未找到 nginx, 请先安装: sudo apt install -y nginx"
    exit 1
fi
info "Nginx: $(nginx -v 2>&1)"

# ---------- 1. 检查站点配置文件 ----------
if [ ! -f "$SRC_CONF" ]; then
    err "站点配置不存在: $SRC_CONF"
    err "请先在 config/nginx/ 下创建 vp.conf"
    exit 1
fi
info "站点配置: $SRC_CONF"

# ---------- 2. 检查目录 ----------
if [ ! -d "$NGINX_SITES_ENABLED" ]; then
    # 兼容 CentOS/RHEL: 用 conf.d 代替 sites-enabled
    if [ -d /etc/nginx/conf.d ]; then
        NGINX_SITES_ENABLED="/etc/nginx/conf.d"
        DEST_LINK="$NGINX_SITES_ENABLED/$SITE_NAME.conf"
        warn "未找到 sites-enabled, 使用 conf.d: $DEST_LINK"
    else
        err "未找到 $NGINX_SITES_ENABLED 或 /etc/nginx/conf.d"
        exit 1
    fi
fi
info "Nginx 加载目录: $NGINX_SITES_ENABLED"

# ---------- 3. 卸载模式 ----------
if [ "$UNINSTALL" = "1" ]; then
    info "卸载模式: 移除软链 $DEST_LINK"
    run "sudo rm -f '$DEST_LINK'"
    info "重载 Nginx"
    run "sudo nginx -t && sudo systemctl reload nginx || sudo nginx -s reload"
    info "完成"
    exit 0
fi

# ---------- 4. 备份并移除默认站点 ----------
DEFAULT_LINK="$NGINX_SITES_ENABLED/default"
if [ -L "$DEFAULT_LINK" ] || [ -f "$DEFAULT_LINK" ]; then
    info "移除默认站点: $DEFAULT_LINK"
    run "sudo rm -f '$DEFAULT_LINK'"
fi

# ---------- 5. 创建软链 (幂等) ----------
if [ -L "$DEST_LINK" ]; then
    CUR_TARGET="$(readlink -f "$DEST_LINK" 2>/dev/null || true)"
    if [ "$CUR_TARGET" = "$SRC_CONF" ]; then
        info "软链已存在且指向正确: $DEST_LINK"
    else
        warn "软链已存在但指向 $CUR_TARGET, 重新创建"
        run "sudo rm -f '$DEST_LINK'"
        run "sudo ln -sf '$SRC_CONF' '$DEST_LINK'"
    fi
else
    info "创建软链: $DEST_LINK -> $SRC_CONF"
    run "sudo ln -sf '$SRC_CONF' '$DEST_LINK'"
fi

# ---------- 6. 语法检查 ----------
info "Nginx 语法检查"
if [ "$DRY_RUN" = "0" ]; then
    if ! sudo nginx -t; then
        err "Nginx 语法检查失败, 请检查 $SRC_CONF"
        exit 1
    fi
fi

# ---------- 7. 启动 / reload ----------
info "启动 / 重载 Nginx"
if [ "$DRY_RUN" = "0" ]; then
    if systemctl is-active --quiet nginx; then
        sudo systemctl reload nginx && info "reload 完成"
    else
        sudo systemctl start nginx && info "启动完成"
        sudo systemctl enable nginx >/dev/null 2>&1 || true
    fi
fi

# ---------- 8. 验证 ----------
info "验证配置已加载"
if [ "$DRY_RUN" = "0" ]; then
    if sudo nginx -T 2>/dev/null | grep -q "listen 80"; then
        info "listen 80 已生效"
    fi
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1/ 2>/dev/null | grep -q "200\|404"; then
        info "HTTP 响应正常"
    else
        warn "HTTP 无响应, 检查 Nginx 是否启动 / 端口是否被占"
    fi
fi

echo ""
info "============================================"
info " Nginx 站点配置完成"
info " 站点配置 : $SRC_CONF"
info " 软链     : $DEST_LINK"
info " 访问     : http://<板卡IP>/"
info "============================================"