#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hw_camera.py — USB MJPG 相机硬件解码封装 (JPU)。

接口故意与 cv2.VideoCapture 对齐, 用于替换 cv2.VideoCapture:
  HwMjpgCamera(device, width, height, fps)
    .read()     -> (ok, bgr_frame)   # JPU 解 NV12 -> 一次 cvtColor 转 BGR
    .get(prop)  -> 支持 WIDTH/HEIGHT/FPS
    .release()
    .isOpened()

底层: ctypes 调 libmjpg_hw.so (V4L2 抓 MJPG 原始帧 + JPU 硬件解码)。
"""
import ctypes
import os
import sys

# ---- 路径注入: config/ + src/ ----
_HERE = os.path.dirname(os.path.abspath(__file__))     # src/
_ROOT = os.path.dirname(_HERE)                         # 项目根
for _p in (os.path.join(_ROOT, 'config'), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

# ---- .so 路径: 优先 libs/, 再退回同目录 / jpu_dec/ ----
_CANDIDATES = (
    os.path.join(_ROOT, 'libs', 'libmjpg_hw.so'),   # 新结构: <root>/libs/
    os.path.join(_HERE, 'libmjpg_hw.so'),           # 兼容: src/
    os.path.join(_HERE, 'jpu_dec', 'libmjpg_hw.so'),# 兼容: src/jpu_dec/
)
_SO = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])

_CAP_W = cv2.CAP_PROP_FRAME_WIDTH if hasattr(cv2, 'CAP_PROP_FRAME_WIDTH') else 3
_CAP_H = cv2.CAP_PROP_FRAME_HEIGHT if hasattr(cv2, 'CAP_PROP_FRAME_HEIGHT') else 4
_CAP_FPS = cv2.CAP_PROP_FPS if hasattr(cv2, 'CAP_PROP_FPS') else 5

_lib = None


def _load():
    """懒加载 libmjpg_hw.so 并声明 C 接口原型（mjcam_open/grab/close），首次调用时执行。"""
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_SO):
        raise RuntimeError(f'libmjpg_hw.so 不存在: {_SO} (需要先编译)')
    lib = ctypes.CDLL(_SO)
    # ---- C 接口 (libmjpg_hw.so) 约定 ----
    #   mjcam_open(dev, w, h, fps) -> void*      打开相机句柄; 失败返回 NULL
    #   mjcam_grab(h, buf, cap, &ow, &oh) -> int 抓一帧并 JPU 硬解为 NV12 写入 buf;
    #                                          返回实际写入字节数 (= w*h*3//2), ow/oh 输出宽高
    #   mjcam_close(h)                           释放相机与解码器资源
    lib.mjcam_open.restype = ctypes.c_void_p
    lib.mjcam_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mjcam_grab.restype = ctypes.c_int
    lib.mjcam_grab.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.mjcam_close.argtypes = [ctypes.c_void_p]
    _lib = lib
    return _lib


class HwMjpgCamera:
    """JPU 硬件解码的 USB MJPG 相机 (cv2.VideoCapture 兼容接口)."""

    def __init__(self, device, width=640, height=480, fps=200):
        """按 device/宽高/帧率打开 JPU 硬解相机，预分配 NV12 帧缓冲。"""
        self._dev = str(device)
        self._w = int(width)
        self._h = int(height)
        self._fps = int(fps)
        self._handle = None
        self._nv12 = None
        self._cap = self._w * self._h * 3 // 2
        try:
            lib = _load()
            h = lib.mjcam_open(self._dev.encode(), self._w, self._h, self._fps)
            if h:
                self._lib = lib
                self._handle = h
                self._nv12 = ctypes.create_string_buffer(self._cap)
                self._ow = ctypes.c_int(0)
                self._oh = ctypes.c_int(0)
        except Exception as exc:
            print(f'[警告] HwMjpgCamera 初始化失败: {exc!r}')
            self._handle = None

    def isOpened(self):
        """返回相机是否成功打开（handle 非空）。"""
        return self._handle is not None

    def _grab(self):
        """抓一帧到内部缓冲; 返回 (ok, nv12_2d_view, ow, oh)."""
        if self._handle is None:
            return False, None, 0, 0
        n = self._lib.mjcam_grab(self._handle, self._nv12, self._cap,
                                 ctypes.byref(self._ow), ctypes.byref(self._oh))
        if n != self._cap:
            return False, None, 0, 0
        w, h = self._ow.value, self._oh.value
        view = np.frombuffer(self._nv12.raw, np.uint8).reshape(h * 3 // 2, w)
        return True, view, w, h

    def read(self):
        """阻塞抓一帧 -> JPU 解 -> BGR。返回 (ok, bgr_frame) 或 (False, None)."""
        ok, nv, _w, _h = self._grab()
        if not ok:
            return False, None
        bgr = cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12)
        return True, bgr

    def grab_raw_nv12(self):
        """抓一帧并保留 NV12 (不转 BGR), 供模型 NV12 直通。"""
        ok, nv, _w, _h = self._grab()
        if not ok:
            return False, None
        return True, nv.copy()

    def read_both(self):
        """抓一帧同时给出 BGR 与 NV12。"""
        ok, nv, _w, _h = self._grab()
        if not ok:
            return False, None, None
        bgr = cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12)
        return True, bgr, nv.copy()

    def get(self, prop_id):
        """读取属性（宽/高/帧率），未打开时返回 0.0。"""
        if self._handle is None:
            return 0.0
        if prop_id == _CAP_W:
            return float(self._ow.value or self._w)
        if prop_id == _CAP_H:
            return float(self._oh.value or self._h)
        if prop_id == _CAP_FPS:
            return float(self._fps)
        return 0.0

    def release(self):
        """关闭相机句柄并释放 JPU 解码资源（幂等）。"""
        if self._handle is not None:
            try:
                self._lib.mjcam_close(self._handle)
            except Exception:
                pass
            self._handle = None