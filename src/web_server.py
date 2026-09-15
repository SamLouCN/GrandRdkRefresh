#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm_reader.py — 读共享内存, Web 服务 / legacy 兼容层用"""
import json
import mmap
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'config'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main_config as MC


class ShmFrameReader:
    """读共享内存最新 JPEG 帧。"""

    def __init__(self, path):
        self.path = path
        self._fd = None
        self._mm = None
        self._size = 0

    def _ensure(self):
        """打开/重开 mmap（写端删除文件后也能重连）。"""
        if self._mm is not None and os.path.exists(self.path):
            return
        if self._mm is not None:
            try:
                self._mm.close()
                os.close(self._fd)
            except Exception:
                pass
            self._fd = self._mm = None
        if not os.path.exists(self.path):
            return
        try:
            self._fd = os.open(self.path, os.O_RDONLY)
            self._size = os.fstat(self._fd).st_size
            self._mm = mmap.mmap(self._fd, self._size, prot=mmap.PROT_READ)
        except Exception:
            self._fd = self._mm = None

    def read_latest(self):
        """返回 (seq, jpeg_bytes) 或 (-1, None)。"""
        self._ensure()
        if self._mm is None:
            return -1, None
        try:
            magic, seq, length, wh, ts_us = struct.unpack(
                MC.HDR_FMT, self._mm[:MC.HDR_SIZE])
            if magic != MC.HDR_MAGIC or length == 0:
                return -1, None
            if length > self._size - MC.HDR_SIZE:
                return -1, None
            jpeg = bytes(self._mm[MC.HDR_SIZE:MC.HDR_SIZE + length])
            return seq, jpeg
        except Exception:
            return -1, None

    def close(self):
        try:
            if self._mm is not None:
                self._mm.close()
            if self._fd is not None:
                os.close(self._fd)
        except Exception:
            pass


class ShmJsonReader:
    """读共享内存 JSON（覆盖写）。"""

    def __init__(self, path):
        self.path = path

    def read(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                raw = f.read().strip()
            if not raw:
                return {}
            return json.loads(raw)
        except Exception:
            return {}