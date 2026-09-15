#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""web_server.py — 读共享内存, 对外提供 MJPEG / REST / WebSocket

数据源:
    /dev/shm/momo_frame_front.bin   (front 最新 JPEG)
    /dev/shm/momo_frame_bottom.bin  (bottom 最新 JPEG)
    /dev/shm/momo_det_front.json    (front 检测结果)
    /dev/shm/momo_det_bottom.json   (bottom 检测结果)
    /dev/shm/momo_stats_front.json  (front 帧率/耗时)
    /dev/shm/momo_stats_bottom.json (bottom 帧率/耗时)

对外接口:
    GET  /cam1              前视 MJPEG 流
    GET  /cam2              下视 MJPEG 流
    GET  /api/status        两路检测 + 统计 (REST)
    GET  /api/detections    两路检测结果 (REST)
    GET  /healthz           健康检查
    WS   /ws/status         两路检测 + 统计 (WebSocket, 默认 10Hz)

部署:
    - 本服务只监听 127.0.0.1:5000
    - 由 Nginx (:80) 反代 /cam1 /cam2 /api/ /ws/ 到本服务
"""
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

# ---- 路径注入: config/ + src/ ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'config'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import main_config as MC
    from shm_reader import ShmFrameReader, ShmJsonReader
except Exception as exc:
    print(f'[FATAL] import 失败: {exc!r}', file=sys.stderr)
    print(f'        PYTHONPATH={os.environ.get("PYTHONPATH", "")}', file=sys.stderr)
    print(f'        sys.path={sys.path}', file=sys.stderr)
    raise

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError as exc:
    print(f'[FATAL] 缺少 Web 依赖: {exc!r}', file=sys.stderr)
    print('        安装: pip3 install fastapi uvicorn websockets', file=sys.stderr)
    raise


# ============================================================
# 启动检查
# ============================================================
_REQUIRED_MC_ATTRS = (
    'SHM_FRAME_FRONT', 'SHM_FRAME_BOTTOM',
    'SHM_DET_FRONT', 'SHM_DET_BOTTOM',
    'SHM_STATS_FRONT', 'SHM_STATS_BOTTOM',
    'HDR_MAGIC', 'HDR_FMT', 'HDR_SIZE',
    'WEB_HOST', 'WEB_PORT', 'WEB_TELEM_HZ',
)

def _check_config():
    missing = [a for a in _REQUIRED_MC_ATTRS if not hasattr(MC, a)]
    if missing:
        raise RuntimeError(
            f'main_config 缺少字段: {missing}\n'
            f'请在 config/main_config.py 中补上'
        )
    print('[*] main_config 字段检查通过', flush=True)
    print(f'[*] SHM_FRAME_FRONT  = {MC.SHM_FRAME_FRONT}',  flush=True)
    print(f'[*] SHM_FRAME_BOTTOM = {MC.SHM_FRAME_BOTTOM}', flush=True)
    print(f'[*] SHM_DET_FRONT    = {MC.SHM_DET_FRONT}',    flush=True)
    print(f'[*] SHM_DET_BOTTOM   = {MC.SHM_DET_BOTTOM}',   flush=True)
    print(f'[*] SHM_STATS_FRONT  = {MC.SHM_STATS_FRONT}',  flush=True)
    print(f'[*] SHM_STATS_BOTTOM = {MC.SHM_STATS_BOTTOM}', flush=True)


_check_config()


# ============================================================
# 共享内存读端
# ============================================================
front_frames  = ShmFrameReader(MC.SHM_FRAME_FRONT)
bottom_frames = ShmFrameReader(MC.SHM_FRAME_BOTTOM)
front_det     = ShmJsonReader(MC.SHM_DET_FRONT)
bottom_det    = ShmJsonReader(MC.SHM_DET_BOTTOM)
front_stats   = ShmJsonReader(MC.SHM_STATS_FRONT)
bottom_stats  = ShmJsonReader(MC.SHM_STATS_BOTTOM)


# ============================================================
# MJPEG 生成器
# ============================================================
MJPEG_BOUNDARY = 'frame'
_POLL_INTERVAL = 0.01          # 无新帧时等待 10ms
_HEARTBEAT_SEC = 2.0           # 超过 2s 没有新帧，发空行保持连接（防 Nginx 断流）

def mjpeg_generator(reader):
    """读共享内存里的最新 JPEG，按 MJPEG 协议输出。

    - seq 不变就不发，避免重复帧占带宽
    - 长时间无新帧时发送空注释行，保持 HTTP 连接不断（Nginx proxy_read_timeout）
    """
    last_seq = -1
    last_send_ts = time.time()
    while True:
        try:
            seq, jpeg = reader.read_latest()
        except Exception:
            seq, jpeg = -1, None

        now = time.time()

        if jpeg is not None and seq != last_seq:
            last_seq = seq
            last_send_ts = now
            yield (b'--' + MJPEG_BOUNDARY.encode() + b'\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                   + jpeg + b'\r\n')
        elif now - last_send_ts > _HEARTBEAT_SEC:
            # 心跳：空注释行，不会破坏 MJPEG 帧
            last_send_ts = now
            yield b'\r\n'

        # 无新帧：让出 CPU
        await_sec = _POLL_INTERVAL
        time.sleep(await_sec)


# ============================================================
# 生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f'[*] web_server 启动: http://{MC.WEB_HOST}:{MC.WEB_PORT}/', flush=True)
    print(f'[*] 端点: /cam1 /cam2 /api/status /api/detections /ws/status', flush=True)
    yield
    print('[*] web_server 停止', flush=True)


app = FastAPI(title='GrandRdk Web Server', lifespan=lifespan)


# ============================================================
# 路由
# ============================================================
@app.get('/cam1')
def cam1():
    """前视 MJPEG 流。"""
    return StreamingResponse(
        mjpeg_generator(front_frames),
        media_type=f'multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close',
        },
    )


@app.get('/cam2')
def cam2():
    """下视 MJPEG 流。"""
    return StreamingResponse(
        mjpeg_generator(bottom_frames),
        media_type=f'multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close',
        },
    )


@app.get('/api/status')
def status():
    """两路检测 + 统计 (REST)。"""
    return {
        'front':  {'det': front_det.read(),  'stats': front_stats.read()},
        'bottom': {'det': bottom_det.read(), 'stats': bottom_stats.read()},
        'ts': time.time(),
    }


@app.get('/api/detections')
def detections():
    """两路检测结果 (REST)。"""
    return {'front': front_det.read(), 'bottom': bottom_det.read()}


@app.get('/healthz')
def healthz():
    """健康检查：确认共享内存文件状态。"""
    def _exists(p):
        try:
            st = os.stat(p)
            return {'exists': True, 'size': st.st_size}
        except FileNotFoundError:
            return {'exists': False, 'size': 0}
        except Exception as e:
            return {'exists': False, 'error': repr(e)}

    return {
        'ok': True,
        'ts': time.time(),
        'shm': {
            'front_frame':  _exists(MC.SHM_FRAME_FRONT),
            'bottom_frame': _exists(MC.SHM_FRAME_BOTTOM),
            'front_det':    _exists(MC.SHM_DET_FRONT),
            'bottom_det':   _exists(MC.SHM_DET_BOTTOM),
            'front_stats':  _exists(MC.SHM_STATS_FRONT),
            'bottom_stats': _exists(MC.SHM_STATS_BOTTOM),
        },
    }


@app.get('/')
def root():
    """本服务根路径：直接访问给个提示（正常走 Nginx）。"""
    return JSONResponse({
        'msg': 'web_server 运行中，请通过 Nginx 访问 http://<board-ip>/',
        'endpoints': ['/cam1', '/cam2', '/api/status', '/api/detections', '/healthz', '/ws/status'],
    })


@app.websocket('/ws/status')
async def ws_status(ws: WebSocket):
    """两路检测 + 统计 (WebSocket)。"""
    await ws.accept()
    interval = 1.0 / max(1, MC.WEB_TELEM_HZ)
    print(f'[ws] client connected: {ws.client}', flush=True)
    try:
        while True:
            try:
                payload = {
                    'front':  {'det': front_det.read(),  'stats': front_stats.read()},
                    'bottom': {'det': bottom_det.read(), 'stats': bottom_stats.read()},
                    'ts': time.time(),
                }
                await ws.send_json(payload)
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f'[ws] send 异常: {e!r}', flush=True)
                break
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
    finally:
        print(f'[ws] client disconnected: {ws.client}', flush=True)


# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    # 打印启动信息
    print('=' * 60, flush=True)
    print('[*] web_server 准备启动', flush=True)
    print(f'[*] host={MC.WEB_HOST} port={MC.WEB_PORT}', flush=True)
    print('=' * 60, flush=True)

    # 检查端口占用（提前报错，避免和别的东西冲突）
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((MC.WEB_HOST, MC.WEB_PORT))
        s.close()
    except OSError as e:
        print(f'[FATAL] 端口 {MC.WEB_PORT} 已被占用: {e!r}', file=sys.stderr, flush=True)
        print(f'        排查: sudo lsof -i :{MC.WEB_PORT}', file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        uvicorn.run(
            app,
            host=MC.WEB_HOST,
            port=MC.WEB_PORT,
            log_level='warning',
            access_log=False,
        )
    except KeyboardInterrupt:
        print('[*] 收到 Ctrl-C, 退出', flush=True)