#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""legacy_bridge.py — 旧协议兼容桥 (读共享内存, 对外 TCP JPEG + UDP 遥测)

对外提供:
  1) TCP JPEG 推流 :9000 (front) / :9001 (bottom)
       协议: struct.pack('<I', len) + JPEG bytes
  2) UDP 遥测 :8081
       10Hz 向 LEGACY_TELEM_DST 单播 $TEL,...#  (41 字段, 与 vp5.1 一致)
       监听 :8081, 收到 b'PING' 回 b'PONG'

数据源: 共享内存
  - /dev/shm/momo_frame_front.bin  (front JPEG)
  - /dev/shm/momo_frame_bottom.bin (bottom JPEG)
"""
import math
import os
import queue
import random
import socket
import struct
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'config'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main_config as MC
from shm_reader import ShmFrameReader


# ==============================================================
# TCP JPEG 推流（协议与旧 streamer.py 完全一致）
# ==============================================================
def _put_latest(q, item):
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class LegacyTcpStreamer(threading.Thread):
    """单端口 TCP JPEG 推流线程（旧 streamer.py 协议）。"""

    def __init__(self, name, port, bind_host, fps, quality, reader):
        super().__init__(daemon=True)
        self.name = name
        self.port = int(port)
        self.bind_host = bind_host
        self.interval = 1.0 / max(1, int(fps))
        self.quality = int(quality)
        self.reader = reader
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._clients = 0
        self._last_seq = -1

    @property
    def connected(self):
        return self._clients > 0

    def stop(self):
        self._stop.set()

    def _pump(self):
        """后台线程: 读共享内存 -> 入队最新帧。"""
        while not self._stop.is_set():
            seq, jpeg = self.reader.read_latest()
            if jpeg is None or seq == self._last_seq:
                time.sleep(0.005)
                continue
            self._last_seq = seq
            if self.connected:
                _put_latest(self._q, jpeg)

    def run(self):
        # 启动 pump 线程
        threading.Thread(target=self._pump, daemon=True).start()

        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.settimeout(0.5)
            srv.bind((self.bind_host, self.port))
            srv.listen(1)
            print(f'[*] legacy[TCP {self.name}]: 监听 {self.bind_host}:{self.port}', flush=True)
        except OSError as exc:
            print(f'[警告] legacy[TCP {self.name}]: 监听失败 {exc!r}', flush=True)
            return

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    self._clients += 1
                    print(f'[*] legacy[TCP {self.name}]: 客户端 {addr[0]}:{addr[1]}', flush=True)
                    conn.settimeout(1.0)
                    try:
                        self._serve(conn)
                    except Exception as exc:
                        print(f'[警告] legacy[TCP {self.name}]: {exc!r}', flush=True)
                    finally:
                        self._clients -= 1
                        print(f'[*] legacy[TCP {self.name}]: 断开 {addr[0]}:{addr[1]}', flush=True)
        finally:
            try:
                srv.close()
            except OSError:
                pass

    def _serve(self, conn):
        next_send = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            wait = next_send - now
            if wait > 0:
                if self._stop.wait(min(wait, 0.05)):
                    return
                now = time.perf_counter()
                if now < next_send:
                    continue
            try:
                jpeg = self._q.get(timeout=0.5)
            except queue.Empty:
                next_send = time.perf_counter() + self.interval
                continue
            try:
                conn.sendall(struct.pack('<I', len(jpeg)) + jpeg)
            except OSError:
                return
            next_send = time.perf_counter() + self.interval


# ==============================================================
# UDP 遥测（协议与旧 telem_sender.py 一致）
# ==============================================================
N_FIELDS = 41


class _Rolling:
    def __init__(self, base, amp, period, noise=0.02, phase=None):
        self.base = base
        self.amp = amp
        self.period = period
        self.noise = noise
        self.phase = phase if phase is not None else random.uniform(0, 6.28)

    def at(self, t):
        v = self.base + self.amp * math.sin(2 * math.pi * t / self.period + self.phase)
        return v + random.uniform(-self.noise, self.noise)


def _build_tel(t):
    actual = [
        _Rolling(2.0, 6.0, 8.0).at(t),
        _Rolling(-1.5, 4.0, 11.0).at(t),
        _Rolling(135.0, 90.0, 25.0).at(t),
        _Rolling(0.0, 20.0, 4.0).at(t),
        _Rolling(0.0, 15.0, 5.0).at(t),
        _Rolling(0.0, 12.0, 6.0).at(t),
        _Rolling(2.5, 0.8, 20.0).at(t),
        _Rolling(0.0, 0.4, 7.0).at(t),
        _Rolling(0.0, 0.3, 9.0).at(t),
        _Rolling(0.0, 0.25, 5.0).at(t),
    ]
    target = [
        _Rolling(2.0, 6.0, 8.0, phase=1.0).at(t),
        _Rolling(-1.5, 4.0, 11.0, phase=1.3).at(t),
        _Rolling(135.0, 90.0, 25.0, phase=1.6).at(t),
        _Rolling(0.0, 20.0, 4.0, phase=0.4).at(t),
        _Rolling(0.0, 15.0, 5.0, phase=0.7).at(t),
        _Rolling(0.0, 12.0, 6.0, phase=1.1).at(t),
        _Rolling(2.5, 0.8, 20.0, phase=0.9).at(t),
        _Rolling(0.0, 0.4, 7.0, phase=1.4).at(t),
        _Rolling(0.0, 0.3, 9.0, phase=0.5).at(t),
        _Rolling(0.0, 0.25, 5.0, phase=2.0).at(t),
    ]
    batt = [
        _Rolling(24.5, 0.3, 60.0).at(t),
        _Rolling(3.2, 1.5, 12.0).at(t),
        _Rolling(87.0, 4.0, 45.0).at(t),
        _Rolling(34.0, 1.5, 40.0).at(t),
        _Rolling(28.0, 1.0, 35.0).at(t),
    ]
    thrusters = [
        max(-1.0, min(1.0, _Rolling(0.0, 0.55, 6.0 + i * 0.7).at(t)))
        for i in range(12)
    ]
    acc = [
        _Rolling(0.0, 0.9, 5.0).at(t),
        _Rolling(0.0, 0.7, 6.0).at(t),
        _Rolling(9.8, 0.4, 30.0).at(t),
        _Rolling(0.0, 0.6, 18.0).at(t),
    ]
    all_vals = actual + target + batt + thrusters + acc
    assert len(all_vals) == N_FIELDS
    body = ",".join(f"{v:.2f}" for v in all_vals)
    return f"$TEL,{body}#\r\n"


def _ping_responder(stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", MC.LEGACY_TELEM_PORT))
    except OSError as exc:
        print(f'[警告] legacy[UDP]: PING 绑定失败 {exc!r}', flush=True)
        return
    s.settimeout(0.5)
    print(f'[*] legacy[UDP]: PING 监听 :{MC.LEGACY_TELEM_PORT}', flush=True)
    while not stop.is_set():
        try:
            data, addr = s.recvfrom(64)
            if data == b"PING":
                s.sendto(b"PONG", addr)
        except socket.timeout:
            continue
        except OSError:
            break
    try:
        s.close()
    except Exception:
        pass


def _telem_sender(stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (MC.LEGACY_TELEM_DST, MC.LEGACY_TELEM_PORT)
    interval = 1.0 / max(0.5, MC.LEGACY_TELEM_HZ)
    print(f'[*] legacy[UDP]: $TEL -> {dst[0]}:{dst[1]} @ {MC.LEGACY_TELEM_HZ}Hz', flush=True)
    t0 = time.time()
    n = 0
    while not stop.is_set():
        frame = _build_tel(time.time() - t0)
        try:
            s.sendto(frame.encode(), dst)
        except OSError:
            pass
        n += 1
        if n % 100 == 0:
            print(f'[*] legacy[UDP]: 已发 {n} 帧', flush=True)
        # 分段 sleep，保证 stop 及时退出
        end = time.time() + interval
        while not stop.is_set() and time.time() < end:
            time.sleep(0.01)
    try:
        s.close()
    except Exception:
        pass


# ==============================================================
# main
# ==============================================================
def main():
    print('[*] legacy_bridge 启动 (旧协议兼容层)', flush=True)

    front_reader  = ShmFrameReader(MC.SHM_FRAME_FRONT)
    bottom_reader = ShmFrameReader(MC.SHM_FRAME_BOTTOM)

    stop = threading.Event()

    # TCP 推流
    front_tcp = LegacyTcpStreamer(
        'front', MC.LEGACY_TCP_FRONT_PORT, MC.LEGACY_TCP_BIND,
        MC.LEGACY_TCP_FPS, MC.LEGACY_TCP_JPEG_Q, front_reader)
    bottom_tcp = LegacyTcpStreamer(
        'bottom', MC.LEGACY_TCP_BOTTOM_PORT, MC.LEGACY_TCP_BIND,
        MC.LEGACY_TCP_FPS, MC.LEGACY_TCP_JPEG_Q, bottom_reader)
    front_tcp.start()
    bottom_tcp.start()

    # UDP 遥测
    if MC.LEGACY_PING_ENABLED:
        threading.Thread(target=_ping_responder, args=(stop,), daemon=True).start()
    threading.Thread(target=_telem_sender, args=(stop,), daemon=True).start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('[*] legacy_bridge 停止', flush=True)
        stop.set()
        front_tcp.stop()
        bottom_tcp.stop()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())