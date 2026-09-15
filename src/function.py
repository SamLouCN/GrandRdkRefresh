"""
function.py — 板端核心库（YOLO 检测 + 水下预处理）

从顶到底两节，建议按需跳读：

    ① YOLO 检测器 (YoloDetector + YoloDetect + 通用辅助)
         - front.py / bottom.py 实际使用的检测入口
         - 适配 hbm (RDK 板端 BPU) / ultralytics (开发机) 两种后端
         - 上层调用: detector = YoloDetector(CFG['FRONT_YOLO']); detector.detect(frame, nv12=...)

    ② 水下图像预处理 (preprocess / _underwater_color_correct / _apply_clahe)
         - 颜色校正 + 高斯去噪 + CLAHE 三个独立开关
         - 被 front.py / bottom.py 复用

外部依赖:
    cv2 / numpy
    main_config.DEFAULT_CONFIG                (来自 config/main_config.py)
    utils.py_utils.preprocess / postprocess   (来自 src/utils/)
"""

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
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from main_config import DEFAULT_CONFIG


# YOLO 预处理/后处理工具 (utils/py_utils)
try:
    import utils.py_utils.preprocess as pre_utils
    import utils.py_utils.postprocess as post_utils
except ImportError:
    pre_utils = post_utils = None
    print("[警告] 未找到 utils/py_utils, YOLO 功能将不可用")


# =====================================================================
# ① YOLO 检测器 (YoloDetector + YoloDetect + 通用辅助)
# =====================================================================

# ======================
# 通用辅助函数
# ======================

def normalize_name(name):
    """标准化类别名, 兼容 大小写/空格/下划线/短横线 差异."""
    return str(name).strip().lower().replace('_', ' ').replace('-', ' ')


def resolve_model_path(model_path):
    """解析模型/标签路径: 相对路径基于本文件所在目录展开."""
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(model_path):
        model_path = os.path.join(_base_dir, model_path)
    return model_path


def load_labels(label_file):
    """读取类别名称文件 (每行一个类名), 未配置/不存在时返回空列表."""
    if not label_file or not os.path.exists(label_file):
        return []
    with open(label_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ======================
# 默认计时器 (未传入 StageTimer 时使用)
# ======================

class NullTimer:
    """空计时器: 关闭阶段统计时使用 (接口与 StageTimer 一致, 近零开销)."""

    def measure(self, stage):
        return _NullMeasure()


class _NullMeasure:
    """空测量上下文管理器：与 NullTimer 配套，近零开销。"""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


NULL_TIMER = NullTimer()


# ======================
# 可视化辅助
# ======================

def draw_detections(frame, detections):
    """在帧上绘制检测结果 (外接框 + 中心点 + 类别置信度)."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"{det['label']} {det['score']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


def format_detection(det, mark_point, idx):
    """格式化单个 YOLO 检测: 标号 + label + 中心点 + 相对相机标识点的偏移."""
    cx, cy = det['center']
    if mark_point:
        mx, my = mark_point
        rel = f"相对标记点=({cx - mx:+d},{cy - my:+d})"
    else:
        rel = '相对标记点=(-,-)'
    return (f"#{idx} {det['label']} 中心=({cx},{cy}) {rel} conf={det['score']:.2f}")


# ======================
# YOLO 检测类
# ======================

@dataclass
class YoloDetectConfig:
    """YoloDetect 模型初始化配置。"""
    model_path: str
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: float = 0.45
    reg: int = 16
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: list = field(default_factory=lambda: [80, 40, 20])


class YoloDetect:
    """基于 HB_HBMRuntime 的 YOLO DFL 检测封装。"""

    def __init__(self, config: YoloDetectConfig, timer=None):
        self.timer = timer or NULL_TIMER
        try:
            import hbm_runtime
        except ImportError as exc:
            raise RuntimeError(
                'hbm 后端需要 RDK 板端环境 (hbm_runtime); '
                '开发机请在配置中将 backend 改为 "ultralytics"'
            ) from exc

        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]

        self.weights_static = np.arange(config.reg, dtype=np.float32)[np.newaxis, np.newaxis, :]
        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                    img: Optional[np.ndarray] = None,
                    image_format: Optional[str] = "BGR",
                    nv12: Optional[np.ndarray] = None,
                    ) -> Dict[str, Dict[str, np.ndarray]]:
        if pre_utils is None:
            raise RuntimeError('utils.py_utils 未加载, 无法执行模型预处理')
        if image_format == "BGR":
            if img is None:
                raise ValueError('image_format=BGR 时必须提供 img')
            resize_img = pre_utils.resized_image(
                img, self.input_w, self.input_h, self.cfg.resize_type)
            y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        elif image_format in ("NV12", "nv12"):
            if nv12 is None:
                raise ValueError('image_format=NV12 时必须提供 nv12 数组')
            ori_h, ori_w = nv12.shape[0] * 2 // 3, nv12.shape[1]
            y, uv = pre_utils.preprocess_nv12(
                nv12, ori_w, ori_h,
                self.input_w, self.input_h)
        else:
            raise ValueError(f"不支持的图像格式: {image_format}")
        if y.ndim == 2:
            y = y[np.newaxis, :, :, np.newaxis]
        if uv.ndim == 3:
            uv = uv[np.newaxis, :, :, :]
        return {
            self.model_name: {
                self.input_names[0]: y,
                self.input_names[1]: uv
            }
        }

    def forward(self, input_tensor):
        return self.model.run(input_tensor)

    def post_process(self,
                     outputs,
                     ori_img_w: int,
                     ori_img_h: int,
                     score_thres: Optional[float] = None,
                     nms_thres: Optional[float] = None,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if post_utils is None:
            raise RuntimeError('utils.py_utils 未加载, 无法执行模型后处理')
        score_thres = score_thres if score_thres is not None else self.cfg.score_thres
        nms_thres = nms_thres if nms_thres is not None else self.cfg.nms_thres

        conf_thres_raw = -np.log(1.0 / score_thres - 1)

        model_outputs = outputs[self.model_name]
        all_boxes = []
        all_scores = []
        all_ids = []
        for i, (stride, anchor_size) in enumerate(
                zip(self.cfg.strides, self.cfg.anchor_sizes)):
            cls_key = self.output_names[2 * i]
            box_key = self.output_names[2 * i + 1]

            scores, ids, valid_indices = post_utils.filter_classification(
                model_outputs[cls_key], conf_thres_raw)

            dbboxes = post_utils.decode_boxes(
                model_outputs[box_key], valid_indices,
                anchor_size, stride, self.weights_static)

            all_boxes.append(dbboxes)
            all_scores.append(scores)
            all_ids.append(ids)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_ids, axis=0)

        keep = post_utils.NMS(boxes, scores, cls_ids, nms_thres)

        xyxy = post_utils.scale_coords_back(
            boxes[keep], ori_img_w, ori_img_h,
            self.input_w, self.input_h, self.cfg.resize_type)

        return xyxy, scores[keep], cls_ids[keep]

    def predict(self,
                img: Optional[np.ndarray] = None,
                image_format: str = "BGR",
                nv12: Optional[np.ndarray] = None,
                score_thres: Optional[float] = None,
                nms_thres: Optional[float] = None,
                ):
        if image_format in ("NV12", "nv12"):
            ori_img_h, ori_img_w = nv12.shape[0] * 2 // 3, nv12.shape[1]
        else:
            ori_img_h, ori_img_w = img.shape[:2]

        with self.timer.measure('YOLO预处理'):
            input_tensor = self.pre_process(img, image_format=image_format, nv12=nv12)

        with self.timer.measure('推理'):
            outputs = self.forward(input_tensor)

        with self.timer.measure('后处理'):
            boxes, scores, cls_ids = self.post_process(
                outputs, ori_img_w, ori_img_h, score_thres, nms_thres)

        return boxes, scores, cls_ids

    def __call__(self, img=None, image_format="BGR", nv12=None,
                 score_thres=None, nms_thres=None):
        return self.predict(img, image_format=image_format, nv12=nv12,
                            score_thres=score_thres, nms_thres=nms_thres)


class YoloDetector:
    """YOLO 检测器适配层: 统一 hbm (RDK 板端 BPU) / ultralytics (开发机) 两种后端."""

    def __init__(self, cfg: dict, timer=None):
        self.cfg = cfg
        self.backend = cfg.get('backend', 'hbm')
        self.timer = timer or NULL_TIMER
        self.preprocess_mode = str(cfg.get('preprocess_mode', 'auto')).lower()
        self.labels = self._resolve_labels(cfg)
        self.target_ids = [int(i) for i in (cfg.get('target_class_ids') or [])]
        names = cfg.get('target_class_names') or []
        self.targets = {normalize_name(n) for n in names} if names else set()
        if self.targets:
            if not self.labels:
                print('[警告] 配置了 target_class_names 但未提供类别名 '
                      '(class_names / label_file), 无法按名称过滤, 将保留全部检测结果')
            else:
                label_set = {normalize_name(l) for l in self.labels}
                missing = [n for n in names if normalize_name(n) not in label_set]
                if missing:
                    raise ValueError(
                        f"target_class_names 中的类别 {missing} 不在类别列表 {self.labels} 中; "
                        '请检查 class_names / label_file 的顺序与模型类别 id 一致')

        if self.backend == 'hbm':
            self._init_hbm()
        elif self.backend == 'ultralytics':
            self._init_ultralytics()
        else:
            raise ValueError(f'未知的检测后端: {self.backend}')

    @staticmethod
    def _resolve_labels(cfg):
        class_names = cfg.get('class_names') or []
        labels = [str(n).strip() for n in class_names if str(n).strip()]
        if labels:
            return labels
        label_file = cfg.get('label_file')
        if label_file:
            return load_labels(resolve_model_path(label_file))
        return []

    def _is_target_cls(self, cls_id, label):
        if self.target_ids:
            return int(cls_id) in self.target_ids
        if self.targets:
            return normalize_name(label) in self.targets
        return True

    def _init_hbm(self):
        model_path = resolve_model_path(self.cfg['model_path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        config = YoloDetectConfig(
            model_path=model_path,
            score_thres=self.cfg['score_thres'],
            nms_thres=self.cfg['nms_thres'],
            strides=self.cfg['strides'],
        )
        self.model = YoloDetect(config, timer=self.timer)
        self.model.cfg.anchor_sizes = [self.model.input_h // s
                                       for s in self.model.cfg.strides]
        self.model.set_scheduling_params(
            priority=self.cfg.get('priority', 0),
            bpu_cores=self.cfg.get('bpu_cores', [0]))

    def _init_ultralytics(self):
        model_path = resolve_model_path(self.cfg['model_path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('未安装 ultralytics, 请先执行: pip install ultralytics') from exc
        self.model = YOLO(model_path)
        if not self.labels and hasattr(self.model, 'names'):
            names = self.model.names
            if isinstance(names, dict):
                self.labels = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                self.labels = list(names)

    def detect(self, frame, nv12=None):
        if frame is None and nv12 is None:
            return []
        if self.backend == 'ultralytics':
            return self._detect_ultralytics(frame)
        return self._detect_hbm(frame, nv12)

    def _detect_hbm(self, frame, nv12=None):
        use_nv12 = (nv12 is not None and self.preprocess_mode in ('auto', 'nv12'))
        if use_nv12:
            boxes, scores, cls_ids = self.model.predict(
                None, image_format='NV12', nv12=nv12)
        else:
            if frame is None and nv12 is not None:
                frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            boxes, scores, cls_ids = self.model.predict(frame)
        return self._format_detections(boxes, scores, cls_ids)

    def _detect_ultralytics(self, frame):
        with self.timer.measure('推理'):
            preds = self.model.predict(frame, conf=self.cfg['score_thres'],
                                       iou=self.cfg['nms_thres'], verbose=False)
        dets = []
        if preds and preds[0].boxes is not None:
            names = preds[0].names
            for box in preds[0].boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                if not self._is_target_cls(cls_id, label):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                dets.append(self._make_detection(label, float(box.conf.item()),
                                                 x1, y1, x2, y2))
        return dets

    def _format_detections(self, boxes, scores, cls_ids):
        dets = []
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            cls_id = int(cls_id)
            label = self.labels[cls_id] if self.labels and cls_id < len(self.labels) else str(cls_id)
            if not self._is_target_cls(cls_id, label):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append(self._make_detection(label, float(score), x1, y1, x2, y2))
        return dets

    @staticmethod
    def _make_detection(label, score, x1, y1, x2, y2):
        return {
            'label': label,
            'score': float(score),
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        }

    def warmup(self, width=640, height=480):
        black = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            self.detect(black)
        except Exception as exc:
            print(f'[警告] 模型预热失败: {exc!r}')


# =====================================================================
# ② 水下图像预处理 (preprocess / _underwater_color_correct / _apply_clahe)
# =====================================================================

def preprocess(frame, config=None):
    """
    通用水下图像预处理: 颜色校正 + 高斯去噪 + CLAHE
    """
    cfg = {
        'enable_color_correct': True,
        'red_boost': 1.2,
        'enable_gaussian': True,
        'gaussian_kernel': 5,
        'enable_clahe': True,
        'clahe_clip': 2.0,
        'clahe_tile': (8, 8),
    }
    if config is not None:
        cfg.update(config)

    result = frame

    if cfg['enable_color_correct']:
        result = _underwater_color_correct(result, cfg['red_boost'])

    if cfg['enable_gaussian']:
        ksize = cfg['gaussian_kernel']
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    if cfg['enable_clahe']:
        result = _apply_clahe(
            result,
            clip_limit=cfg['clahe_clip'],
            tile_grid_size=cfg['clahe_tile'],
        )

    return result


def _underwater_color_correct(frame, red_boost=1.2):
    """水下颜色校正: 灰度世界白平衡 + 红色通道增强 (LUT 加速)."""
    mean_b, mean_g, mean_r, _ = cv2.mean(frame)
    gray_mean = (mean_b + mean_g + mean_r) / 3.0

    def _gain(m):
        return gray_mean / m if m > 1e-6 else 1.0

    gb, gg, gr = _gain(mean_b), _gain(mean_g), _gain(mean_r) * red_boost

    lut = np.empty((256, 1, 3), dtype=np.uint8)
    lut[:, 0, 0] = np.clip(np.arange(256) * gb, 0, 255)
    lut[:, 0, 1] = np.clip(np.arange(256) * gg, 0, 255)
    lut[:, 0, 2] = np.clip(np.arange(256) * gr, 0, 255)
    return cv2.LUT(frame, lut)


def _apply_clahe(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    """LAB 空间 CLAHE 自适应直方图均衡化."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)