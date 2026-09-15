#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm_writer.py — 写共享内存(帧 + JSON), 检测进程用"""
import json
import mmap
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'config'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main_config as MC


class ShmFrameWriter:
    """写最新 JPEG 到共享内存。定长容量, 覆盖式。"""

    def __init__(self, path, width=640, height=480, max_jpeg=None):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.max_jpeg = int(max_jpeg if max_jpeg is not None else MC.MAX_JPEG_BYTES)
        self.total = MC.HDR_SIZE + self.max_jpeg
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.truncate(self.total)
        self._fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, self.total)
        self._seq = 0

    def write(self, jpeg):
        """写入一帧 JPEG。"""
        if not jpeg or len(jpeg) > self.max_jpeg:
            return
        self._seq += 1
        hdr = struct.pack(MC.HDR_FMT, MC.HDR_MAGIC, self._seq,
                          len(jpeg), self.width * self.height,
                          int(time.time() * 1e6))
        self._mm[:MC.HDR_SIZE] = hdr
        self._mm[MC.HDR_SIZE:MC.HDR_SIZE + len(jpeg)] = jpeg
        self._mm.flush()

    def close(self):
        try:
            self._mm.close()
            os.close(self._fd)
        except Exception:
            pass


class ShmJsonWriter:
    """写 JSON 到共享内存(定长, 覆盖式)。"""

    def __init__(self, path, max_bytes=None):
        self.path = path
        self.max_bytes = int(max_bytes if max_bytes is not None else MC.MAX_JSON_BYTES)
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.truncate(self.max_bytes)
        self._fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, self.max_bytes)

    def write(self, obj):
        """写入 JSON 对象（覆盖式）。"""
        data = json.dumps(obj, ensure_ascii=False, default=str).encode('utf-8')
        if len(data) > self.max_bytes:
            data = data[:self.max_bytes]
        pad = self.max_bytes - len(data)
        self._mm.seek(0)
        self._mm.write(data)
        if pad > 0:
            self._mm.write(b' ' * pad)
        self._mm.flush()

    def close(self):
        try:
            self._mm.close()
            os.close(self._fd)
        except Exception:
            pass