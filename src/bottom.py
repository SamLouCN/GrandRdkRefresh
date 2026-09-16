#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bottom.py — 下视任务入口（真实/虚拟相机 + YOLO 检测输出）

数据流:
  camera/虚拟帧 --producer--> q --worker x N--> YOLO 结果
                                     |
                                     |--> 终端 / logs/bottom.log
                                     |--> 帧: /dev/shm/momo_frame_bottom.bin (JPEG)
                                     |--> 检测: /dev/shm/momo_det_bottom.json
                                     |--> 统计: /dev/shm/momo_stats_bottom.json
                                     |--> 光流: /dev/shm/momo_flow_bottom.bin (NV12, 可选)
                                     |--> imshow（SHOW=True 时）
"""
import argparse
import os
import queue
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'config'),
           _HERE,
           os.path.join(_HERE, 'utils')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

import main_config as MC
from main_config import DEFAULT_CONFIG as CFG
from function import YoloDetector, format_detection, draw_detections
from shm_writer import ShmFrameWriter, ShmJsonWriter


TASK = 'bottom'
CAM = CFG['BOTTOM_CAMERA']
YOLO_CFG = CFG['BOTTOM_YOLO']

FLOW_SHARE_ON = bool(CFG.get('ENABLE_FLOW_SHARE_BOTTOM', False))

W = int(CAM.get('width', 640))
H = int(CAM.get('height', 480))
FPS = max(1, min(int(CAM.get('fps', 60)), 60))
MARK_POINT = CAM.get('mark_point')
Q_SIZE = int(CFG.get('CAMERA_QUEUE_SIZE', 8))
LOOP_SLEEP = float(CFG.get('LOOP_SLEEP', 0.0))
SHOW = bool(CFG.get('SHOW', True))
N_WORKERS = int(CFG.get('N_WORKERS', 3))
TIMING = bool(CFG.get('ENABLE_TIMING', False))
TIMING_INTERVAL = int(CFG.get('TIMING_INTERVAL', 30))
SIMPLE_TIMING = bool(CFG.get('SIMPLE_TIMING', False))
LOG_ENABLED = bool(CFG.get('ENABLE_LOG', False))
LOG_DIR = str(CFG.get('LOG_DIR', 'logs'))

CAM_ENABLED = bool(CFG.get('ENABLE_BOTTOM_CAM', True))
YOLO_ENABLED = bool(CFG.get('ENABLE_BOTTOM_YOLO', True))

PRE_MODE = str(YOLO_CFG.get('preprocess_mode', 'auto')).lower()


# ============================================================
# FrameLog / StageStats / format_timing  — 同 front.py
# ============================================================
class FrameLog:
    def __init__(self, path, t_start_perf):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fp = open(path, 'a', encoding='utf-8')
        self.lock = threading.Lock()
        self.t_start_perf = t_start_perf

    def write(self, frame_id, dets, status='done'):
        ts_str = time.strftime('%H:%M:%S')
        elapsed = time.perf_counter() - self.t_start_perf
        if status == 'dropped':
            result = 'dropped(queue_full)'
        elif status == 'no_yolo':
            result = 'no_yolo'
        elif dets:
            result = '; '.join(
                f"{d['label']} conf={d['score']:.2f} "
                f"bbox={tuple(d['bbox'])} center={tuple(d['center'])}"
                for d in dets)
        else:
            result = 'none'
        line = f'[{ts_str}][{elapsed:.3f}s][frame:{frame_id}] [{status}]: {result}\n'
        with self.lock:
            self.fp.write(line)
            self.fp.flush()

    def close(self):
        with self.lock:
            try:
                self.fp.close()
            except Exception:
                pass


class StageStats:
    def __init__(self, simple=False):
        self.lock = threading.Lock()
        self.simple = bool(simple)
        self.capture_s = 0.0
        self.detect_s = 0.0
        self.display_s = 0.0
        self.frames = 0
        self.capture_frames = 0
        self.detect_frames = 0
        self.display_frames = 0
        self._base_capture = 0.0
        self._base_detect = 0.0
        self._base_display = 0.0
        self._base_capture_f = 0
        self._base_detect_f = 0
        self._base_display_f = 0

    def add(self, stage, dt):
        with self.lock:
            if stage == 'capture':
                if not self.simple:
                    self.capture_s += dt
                self.capture_frames += 1
            elif stage == 'detect':
                if not self.simple:
                    self.detect_s += dt
                self.detect_frames += 1
            elif stage == 'display':
                if not self.simple:
                    self.display_s += dt
                self.display_frames += 1
            self.frames += 1

    def mark_window(self):
        with self.lock:
            self._base_capture = self.capture_s
            self._base_detect = self.detect_s
            self._base_display = self.display_s
            self._base_capture_f = self.capture_frames
            self._base_detect_f = self.detect_frames
            self._base_display_f = self.display_frames

    def delta_window(self):
        with self.lock:
            return (
                self.capture_s - self._base_capture,
                self.detect_s - self._base_detect,
                self.display_s - self._base_display,
                self.capture_frames - self._base_capture_f,
                self.detect_frames - self._base_detect_f,
                self.display_frames - self._base_display_f,
            )


def format_timing(elapsed, interval_frames, d_capture, d_detect, d_display, title):
    if interval_frames <= 0 or elapsed <= 0:
        return ''
    fps = interval_frames / elapsed
    ms = lambda s: s / interval_frames * 1000.0
    return (f'[{TASK}] {title}: 帧数={interval_frames} 运行={elapsed:.2f}s '
            f'平均帧率={fps:.1f}fps | '
            f'采集={ms(d_capture):.2f}ms/帧 '
            f'检测={ms(d_detect):.2f}ms/帧 '
            f'显示={ms(d_display):.2f}ms/帧')


def make_frame(frame_id):
    """合成测试帧：动态渐变背景 + 移动黄色圆（对应 yellow-ball 目标）。"""
    t = frame_id
    img = np.zeros((H, W, 3), dtype=np.uint8)
    col = np.arange(W, dtype=np.float32) / W
    img[:, :, 0] = (90 + 60 * np.sin(t * 0.03)).astype(np.uint8)
    img[:, :, 1] = (col * 80 + 50).astype(np.uint8)
    img[:, :, 2] = (col * 40 + 40).astype(np.uint8)
    x = int(W - 40 - ((t * 3) % (W - 80)))
    y = int(H // 3 + 50 * np.sin(t * 0.04 + 1.0))
    cv2.circle(img, (x, y), 36, (0, 255, 255), -1)
    cv2.putText(img, f'{TASK} #{t}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def open_camera(device, index):
    if CAM.get('hardware_decode', False):
        dev = device if device is not None else '/dev/video%d' % (index if index is not None else 0)
        try:
            from hw_camera import HwMjpgCamera
            cam = HwMjpgCamera(dev, W, H, int(CAM.get('fps', 200)))
            if cam.isOpened():
                print(f'[*] [{TASK}] 相机 {dev}: 启用 JPU 硬件解码', flush=True)
                return cam
            print(f'[警告] [{TASK}] 相机 {dev}: JPU 硬件解码不可用, 回退 cv2 软解', flush=True)
        except Exception as exc:
            print(f'[警告] [{TASK}] 相机硬件解码初始化异常 {exc!r}, 回退 cv2 软解', flush=True)
    src = device if device is not None else index
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fmt = CAM.get('format')
    if fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    fps_req = CAM.get('fps')
    if fps_req:
        cap.set(cv2.CAP_PROP_FPS, int(fps_req))
    try:
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
    except cv2.error:
        pass
    return cap


def producer(cap, virtual, q, max_frames, stop, n_workers, stats, enable_timing,
             log_writer, stats_w, prefer_nv12=False, want_bgr=True,
             flow_share_w=None):
    """采集线程：读帧 -> 入队；周期性写 stats；若 flow_share_w 存在则同时写光流共享帧。"""
    fid = 0
    t_win = time.perf_counter()
    win_frames = 0
    fail_cnt = 0
    READ_FAIL_LIMIT = 5
    try:
        while not stop.is_set():
            if max_frames and fid >= max_frames:
                break
            t0 = time.perf_counter()
            nv12 = None
            if virtual:
                frame = make_frame(fid)
            elif prefer_nv12:
                if want_bgr:
                    ok, frame, nv12 = cap.read_both()
                else:
                    ok, nv12 = cap.grab_raw_nv12()
                    frame = None
                if not ok or nv12 is None:
                    fail_cnt += 1
                    if log_writer is not None:
                        log_writer.write(fid, [], status='no_data')
                    if fail_cnt >= READ_FAIL_LIMIT:
                        print(f'[{TASK}] 相机掉线/无数据，本进程退出', flush=True)
                        stop.set()
                        break
                    time.sleep(0.1)
                    fid += 1
                    continue
                fail_cnt = 0
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_cnt += 1
                    if log_writer is not None:
                        log_writer.write(fid, [], status='no_data')
                    if fail_cnt >= READ_FAIL_LIMIT:
                        print(f'[{TASK}] 相机掉线/无数据，本进程退出', flush=True)
                        stop.set()
                        break
                    time.sleep(0.1)
                    fid += 1
                    continue
                fail_cnt = 0
            t1 = time.perf_counter()

            # ---- 写光流共享帧（NV12 优先）----
            if flow_share_w is not None:
                try:
                    flow_share_w.write(nv12 if nv12 is not None else frame)
                except Exception:
                    pass

            if enable_timing:
                stats.add('capture', t1 - t0)
            try:
                q.put((fid, frame, nv12), timeout=1.0)
            except queue.Full:
                if log_writer is not None:
                    log_writer.write(fid, [], status='dropped')
            fid += 1
            win_frames += 1

            # ---- 周期性写 stats ----
            if win_frames >= TIMING_INTERVAL:
                now = time.perf_counter()
                elapsed = now - t_win
                fps_val = win_frames / max(elapsed, 1e-9)
                d_c, d_d, d_dp, n_c, n_d, n_dp = stats.delta_window()
                cap_ms = (d_c / n_c * 1000.0) if n_c else 0.0
                det_ms = (d_d / n_d * 1000.0) if n_d else 0.0
                if stats_w is not None:
                    try:
                        stats_w.write({
                            'fps': round(fps_val, 2),
                            'capture_ms': round(cap_ms, 2),
                            'detect_ms': round(det_ms, 2),
                            'frames': fid,
                            'ts': time.time(),
                        })
                    except Exception:
                        pass
                if enable_timing:
                    if SIMPLE_TIMING:
                        print(f'[{TASK}] 简化时间统计@{fid}帧: 帧数={win_frames} '
                              f'运行={elapsed:.2f}s 平均帧率={fps_val:.1f}fps', flush=True)
                    else:
                        print(format_timing(elapsed, win_frames, d_c, d_d, d_dp,
                                            f'时间统计@{fid}帧'), flush=True)
                stats.mark_window()
                t_win = now
                win_frames = 0

            if virtual:
                time.sleep(1.0 / FPS)
            elif LOOP_SLEEP > 0:
                time.sleep(LOOP_SLEEP)
    finally:
        for _ in range(n_workers):
            q.put(None)


def worker(wid, q, disp_q, show, stop, stats, enable_timing, log_writer,
           frame_w, det_w):
    """检测 worker：检测 -> 写帧/检测结果到共享内存。"""
    detector = YoloDetector(YOLO_CFG) if YOLO_ENABLED else None
    idx = 0
    while not stop.is_set():
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break
        fid, frame, nv12 = item
        status = 'no_yolo' if detector is None else 'done'
        if enable_timing:
            t0 = time.perf_counter()
            dets = detector.detect(frame, nv12=nv12) if detector is not None else []
            t1 = time.perf_counter()
            stats.add('detect', t1 - t0)
        else:
            dets = detector.detect(frame, nv12=nv12) if detector is not None else []

        if log_writer is not None:
            log_writer.write(fid, dets, status=status)

        # ---- 准备可视帧（BGR）----
        if frame is None and nv12 is not None:
            frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        if frame is not None:
            vis = draw_detections(frame, dets) if dets else frame

            # ---- 写帧到共享内存 ----
            if frame_w is not None:
                ok, buf = cv2.imencode(
                    '.jpg', vis,
                    [cv2.IMWRITE_JPEG_QUALITY, MC.WEB_MJPEG_QUALITY])
                if ok:
                    try:
                        frame_w.write(buf.tobytes())
                    except Exception:
                        pass

            # ---- 写检测结果 JSON ----
            if det_w is not None:
                try:
                    det_w.write({
                        'frame': fid,
                        'ts': time.time(),
                        'dets': dets,
                    })
                except Exception:
                    pass

            # ---- imshow ----
            if show:
                t2 = time.perf_counter()
                try:
                    disp_q.put_nowait(vis)
                except queue.Full:
                    pass
                t3 = time.perf_counter()
                if enable_timing:
                    stats.add('display', t3 - t2)

        for d in dets:
            print(f'[{TASK}] ' + format_detection(d, MARK_POINT, idx), flush=True)
            idx += 1
    q.task_done()


def display_loop(disp_q, stop):
    win = TASK
    try:
        while not stop.is_set():
            try:
                vis = disp_q.get(timeout=0.1)
            except queue.Empty:
                continue
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop.set()
                break
    except cv2.error:
        pass
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass


def main():
    ap = argparse.ArgumentParser(description='bottom 任务（真实/虚拟相机 + YOLO）')
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--index', type=int, default=None)
    ap.add_argument('--virtual', action='store_true')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--no-timing', action='store_true')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--no-log', action='store_true')
    args = ap.parse_args()

    enable_timing = TIMING
    if args.timing:
        enable_timing = True
    if args.no_timing:
        enable_timing = False
    enable_log = LOG_ENABLED
    if args.log:
        enable_log = True
    if args.no_log:
        enable_log = False

    if not CAM_ENABLED and not args.virtual:
        print(f'[{TASK}] ENABLE_BOTTOM_CAM=False，跳过真实相机；可用 --virtual 联调')
        return 0

    show = SHOW and not args.no_show

    stats = StageStats(simple=SIMPLE_TIMING)
    stats.mark_window()
    t_start = time.perf_counter()

    log_writer = None
    if enable_log:
        log_writer = FrameLog(os.path.join(LOG_DIR, f'{TASK}.log'), t_start)

    # ---- 共享内存写端 ----
    print(f'[*] [{TASK}] 初始化共享内存...', flush=True)
    frame_w = ShmFrameWriter(MC.SHM_FRAME_BOTTOM, W, H)
    det_w   = ShmJsonWriter(MC.SHM_DET_BOTTOM)
    stats_w = ShmJsonWriter(MC.SHM_STATS_BOTTOM)
    print(f'[*] [{TASK}] 帧共享 : {MC.SHM_FRAME_BOTTOM}', flush=True)
    print(f'[*] [{TASK}] 检测共享: {MC.SHM_DET_BOTTOM}', flush=True)
    print(f'[*] [{TASK}] 统计共享: {MC.SHM_STATS_BOTTOM}', flush=True)

    # ---- 光流共享帧 ----
    flow_share_w = None
    if FLOW_SHARE_ON and not args.virtual:
        try:
            from flow_share import FlowShareWriter
            flow_share_w = FlowShareWriter('bottom', W, H, fmt=0)
            print(f'[*] [{TASK}] 光流共享: {MC.SHM_FLOW_BOTTOM} (NV12)', flush=True)
        except Exception as exc:
            print(f'[警告] [{TASK}] 光流共享初始化失败: {exc!r}', flush=True)

    cap = None
    if not args.virtual:
        device = args.device if args.device is not None else CAM.get('device')
        index = args.index if args.index is not None else CAM.get('index')
        cap = open_camera(device, index)
        if cap is None:
            print(f'[{TASK}] 真实相机打开失败: device={device} index={index}，进程退出', flush=True)
            frame_w.close(); det_w.close(); stats_w.close()
            return 1

    src_is_hw = cap is not None and hasattr(cap, 'grab_raw_nv12') and hasattr(cap, 'read_both')
    prefer_nv12 = bool(src_is_hw and PRE_MODE in ('auto', 'nv12'))
    want_bgr = True   # 需要写共享内存

    q = queue.Queue(maxsize=Q_SIZE)
    disp_q = queue.Queue(maxsize=2)
    stop = threading.Event()

    threads = [threading.Thread(
        target=producer,
        args=(cap, args.virtual, q, args.frames, stop, args.workers,
              stats, enable_timing, log_writer, stats_w,
              prefer_nv12, want_bgr, flow_share_w),
        daemon=True)]
    for i in range(args.workers):
        threads.append(threading.Thread(
            target=worker,
            args=(i, q, disp_q, show, stop, stats, enable_timing, log_writer,
                  frame_w, det_w),
            daemon=True))
    if show:
        threads.append(threading.Thread(
            target=display_loop, args=(disp_q, stop), daemon=True))

    for th in threads:
        th.start()
    try:
        for th in threads:
            th.join()
    except KeyboardInterrupt:
        stop.set()
    finally:
        if cap is not None:
            cap.release()
        if log_writer is not None:
            log_writer.close()
        frame_w.close()
        det_w.close()
        stats_w.close()
        if enable_timing:
            elapsed = time.perf_counter() - t_start
            total = max(stats.capture_frames, 1)
            print(f'[{TASK}] 结束统计: 总运行={elapsed:.2f}s '
                  f'总帧数={stats.capture_frames} 总检测帧={stats.detect_frames} '
                  f'平均帧率={total / elapsed:.1f}fps', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())