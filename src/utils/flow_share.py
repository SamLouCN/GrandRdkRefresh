#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# flow_share.py — 相机帧共享桥 (vp5.x producer -> 光流测速进程)
# ============================================================
# 目的: 让 vp5.x 保持独占相机 + YOLO 帧率的前提下, 把解出的帧
#       共享给独立的光流测速进程 (momo_pwmnet/flow_speed.py), 免第三相机。
# 机制:
#   - 单文件共享内存 /dev/shm/momo_flow_<name>.bin
#   - 布局: [header 48B][数据区 w*h*3 字节]
#   - header: magic 'MFS1', ver, seq, width, height, fmt(0=NV12 1=BGR), ts_us
#   - 并发: fcntl.flock 互斥 (写端 EX, 读端 SH), 无撕裂; 写一次 ~0.1ms,
#           对 200fps 采集的额外开销 <2% 单核
# 写端 (vp5.1 front.py/bottom.py producer):
#   sh = FlowShareWriter('front', 640, 480, fmt=0)   # 进程内建一次
#   sh.write(nv12_or_bgr)                             # 每帧 read 后调用
# 读端 (momo_pwmnet/flow_speed.py):
#   rd = FlowShareReader('front')
#   seq, img = rd.read()          # img: NV12 (h*3//2, w) 或 BGR (h,w,3)
# ============================================================

import fcntl
import mmap
import os
import struct
import time

MAGIC = b"MFS1"
HDR = struct.Struct("<4sIIIII")        # magic, ver, seq, w, h, fmt
HDR_TS = struct.Struct("<Q")           # ts_us
_HEADER_SIZE = HDR.size + HDR_TS.size  # 48 字节
FMT_NV12 = 0
FMT_BGR = 1


def _path(name):
    return f"/dev/shm/momo_flow_{name}.bin"


class FlowShareWriter:
    """写端: 每次 write() 覆盖最新一帧 (持 EX 锁, 单缓冲)。"""

    def __init__(self, name, width, height, fmt=FMT_NV12, create=True):
        self.name = name
        self.w, self.h = int(width), int(height)
        self.fmt = fmt
        data_bytes = self.w * self.h * 3          # 统一按 BGR 最大容量分配
        self._total = _HEADER_SIZE + data_bytes
        self._fh = open(_path(name), "w+b")
        if create:
            self._fh.truncate(self._total)        # 首次建文件时定型
            self._fh.flush()
        self._mm = mmap.mmap(self._fh.fileno(), self._total)
        if create:                                 # 初始化 header
            HDR.pack_into(self._mm, 0, MAGIC, 1, 0, self.w, self.h, self.fmt)
            HDR_TS.pack_into(self._mm, HDR.size, 0)
            self._mm.flush()

    def write(self, frame):
        """frame: uint8 数组, NV12 为 (h*3//2, w), BGR 为 (h, w, 3)。返回 seq。"""
        if frame is None or frame.size == 0:
            return None
        n = int(frame.size)
        if n > self._total - _HEADER_SIZE:
            return None
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        try:
            # 读当前 seq, 递增
            seq = HDR.unpack_from(self._mm, 0)[2] + 1
            # 数据区 (可能 < 容量, 只拷实际长度)
            self._mm[_HEADER_SIZE:_HEADER_SIZE + n] = frame.tobytes()
            # header 最后更新: seq/时间戳 (数据写完后读者才见到新 seq)
            HDR.pack_into(self._mm, 0, MAGIC, 1, seq, self.w, self.h, self.fmt)
            HDR_TS.pack_into(self._mm, HDR.size, int(time.time() * 1e6))
            self._mm.flush()
            return seq
        finally:
            fcntl.flock(self._fh, fcntl.LOCK_UN)

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()


class FlowShareReader:
    """读端: read() 返回 (seq, frame); 无新帧时返回 (last_seq, None)。"""

    def __init__(self, name, timeout_s=5.0):
        self.name = name
        self.path = _path(name)
        self._fh = None
        self._mm = None
        self._total = 0
        self._w = self._h = self._fmt = 0
        deadline = time.time() + timeout_s
        while self._mm is None and time.time() < deadline:  # 等写端建好文件
            try:
                self._fh = open(self.path, "r+b")
                # 读 header 得到尺寸
                self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
                magic, ver, seq, w, h, fmt = HDR.unpack_from(self._mm, 0)
                if magic != MAGIC:
                    raise RuntimeError(f"bad magic: {magic}")
                self._total = _HEADER_SIZE + w * h * 3
                self._w, self._h, self._fmt = w, h, fmt
                self._last_seq = -1
                return
            except (FileNotFoundError, RuntimeError):
                time.sleep(0.2)
        raise RuntimeError(f"flow_share '{name}' 不可用: {self.path}")

    def _mm_ro(self):
        return self._mm

    def read(self, last_seq=None):
        """返回 (seq, frame): frame 为最新完整帧; 无更新返回 (last, None)。"""
        if last_seq is None:
            last_seq = self._last_seq if hasattr(self, "_last_seq") else -1
        fcntl.flock(self._fh, fcntl.LOCK_SH)
        try:
            magic, ver, seq, w, h, fmt = HDR.unpack_from(self._mm, 0)
            if magic != MAGIC or seq <= last_seq:
                return seq if magic == MAGIC else last_seq, None
            if fmt == FMT_NV12:
                n = w * h * 3 // 2
                buf = self._mm[_HEADER_SIZE:_HEADER_SIZE + n]
                import numpy as np
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h * 3 // 2, w)
            else:
                import numpy as np
                buf = self._mm[_HEADER_SIZE:_HEADER_SIZE + w * h * 3]
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            self._last_seq = seq
            return seq, frame
        finally:
            fcntl.flock(self._fh, fcntl.LOCK_UN)

    def close(self):
        if self._mm is not None:
            self._mm.close()
        if self._fh is not None:
            self._fh.close()
