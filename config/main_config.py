import os
import struct

import quick_config as QC


# ======================
# 项目路径
# ======================
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

PATH_CONFIG = os.path.join(PROJECT_ROOT, 'config')
PATH_SRC    = os.path.join(PROJECT_ROOT, 'src')
PATH_WEB    = os.path.join(PROJECT_ROOT, 'web')
PATH_MODELS = os.path.join(PROJECT_ROOT, 'models')
PATH_LIBS   = os.path.join(PROJECT_ROOT, 'libs')
PATH_LOGS   = os.path.join(PROJECT_ROOT, 'logs')

os.makedirs(PATH_LOGS, exist_ok=True)

if os.path.isabs(QC.YOLO_MODEL):
    YOLO_MODEL = QC.YOLO_MODEL
else:
    YOLO_MODEL = os.path.join(PATH_MODELS, QC.YOLO_MODEL)

JPU_LIB = os.path.join(PATH_LIBS, 'libmjpg_hw.so')

# ======================
# 共享内存
# ======================
SHM_DIR          = '/dev/shm'
SHM_FRAME_FRONT  = f'{SHM_DIR}/momo_frame_front.bin'
SHM_FRAME_BOTTOM = f'{SHM_DIR}/momo_frame_bottom.bin'
SHM_DET_FRONT    = f'{SHM_DIR}/momo_det_front.json'
SHM_DET_BOTTOM   = f'{SHM_DIR}/momo_det_bottom.json'
SHM_TELEM        = f'{SHM_DIR}/momo_telemetry.json'
SHM_FLOW_BOTTOM  = f'{SHM_DIR}/momo_flow_bottom.bin'

SHM_STATS_FRONT  = f'{SHM_DIR}/momo_stats_front.json'
SHM_STATS_BOTTOM = f'{SHM_DIR}/momo_stats_bottom.json'

HDR_MAGIC = b'MFS1'
HDR_FMT   = '<4sIIIQ'
HDR_SIZE  = struct.calcsize(HDR_FMT)

MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024

# ======================
# Web 服务
# ======================
WEB_HOST = '127.0.0.1'
WEB_PORT = 5000
WEB_MJPEG_QUALITY = 80
WEB_TELEM_HZ = 10

# ======================
# Web 后端
# ======================
ENABLE_WEB_NEW      = True
ENABLE_WEB_LEGACY   = True

LEGACY_TCP_FRONT_PORT  = 9000
LEGACY_TCP_BOTTOM_PORT = 9001
LEGACY_TCP_BIND        = '0.0.0.0'
LEGACY_TCP_FPS         = 60
LEGACY_TCP_JPEG_Q      = 80

LEGACY_TELEM_PORT      = 8081
LEGACY_TELEM_DST       = '192.168.127.100'
LEGACY_TELEM_HZ        = 10.0
LEGACY_PING_ENABLED    = True

DEFAULT_CONFIG = {

# ======================
# 全局开关（来自 quick_config）
# ======================
    'ENABLE_FRONT_CAM':  QC.ENABLE_FRONT,
    'ENABLE_BOTTOM_CAM': QC.ENABLE_BOTTOM,
    'ENABLE_FLOW_SHARE': False,
    'ENABLE_FLOW_SHARE_BOTTOM': True,

    'ENABLE_FRONT_YOLO':  QC.ENABLE_FRONT_YOLO,
    'ENABLE_BOTTOM_HSV':  False,
    'ENABLE_BOTTOM_YOLO': QC.ENABLE_BOTTOM_YOLO,

    'ENABLE_PREPROCESS_FRONT':  False,
    'ENABLE_PREPROCESS_BOTTOM': False,

# ======================
# 前视相机
# ======================
    'FRONT_CAMERA': {
        'device': QC.FRONT_DEVICE,
        'index':  0,
        'width':  640,
        'height': 480,
        'fps':    QC.FRONT_FPS,
        'format': 'MJPG',
        'backend': 'auto',
        'hardware_decode': True,
        'mark_point': [320, 240],
    },

    'FRONT_PREPROCESS': {
        'enable_color_correct': True,
        'red_boost': 1.2,
        'enable_gaussian': False,
        'gaussian_kernel': 5,
        'enable_clahe': False,
        'clahe_clip': 2.0,
        'clahe_tile': (8, 8),
    },

    'FRONT_YOLO': {
        'backend': 'hbm',
        'preprocess_mode': 'auto',
        'model_path': YOLO_MODEL,
        'score_thres': QC.SCORE_THRES,
        'nms_thres':   QC.NMS_THRES,
        'strides': [8, 16, 32],
        'priority': 0,
        'bpu_cores': QC.FRONT_BPU_CORES,
        'class_names': QC.CLASS_NAMES,
        'label_file': None,
        'target_class_names': QC.FRONT_TARGETS,
        'target_class_ids': [],
    },

# ======================
# 下视相机
# ======================
    'BOTTOM_CAMERA': {
        'device': QC.BOTTOM_DEVICE,
        'index':  2,
        'width':  640,
        'height': 480,
        'fps':    QC.BOTTOM_FPS,
        'format': 'MJPG',
        'backend': 'auto',
        'hardware_decode': True,
        'mark_point': [320, 240],
    },

    'BOTTOM_PREPROCESS': {
        'enable_color_correct': True,
        'red_boost': 1.2,
        'enable_gaussian': False,
        'gaussian_kernel': 5,
        'enable_clahe': False,
        'clahe_clip': 2.0,
        'clahe_tile': (8, 8),
    },

    'BOTTOM_YOLO': {
        'backend': 'hbm',
        'preprocess_mode': 'auto',
        'model_path': YOLO_MODEL,
        'score_thres': QC.SCORE_THRES,
        'nms_thres':   QC.NMS_THRES,
        'strides': [8, 16, 32],
        'priority': 0,
        'bpu_cores': QC.BOTTOM_BPU_CORES,
        'class_names': QC.CLASS_NAMES,
        'label_file': None,
        'target_class_names': QC.BOTTOM_TARGETS,
        'target_class_ids': [],
    },

# ======================
# 主循环 / 采集
# ======================
    'LOOP_SLEEP': 0.001,
    'N_WORKERS': QC.N_WORKERS,
    'CAMERA_QUEUE_SIZE': QC.CAMERA_QUEUE_SIZE,

    'ENABLE_TIMING': QC.ENABLE_TIMING,
    'TIMING_INTERVAL': 100,
    'SIMPLE_TIMING': QC.SIMPLE_TIMING,

    'ENABLE_LOG': QC.ENABLE_LOG,
    'LOG_DIR': PATH_LOGS,

    'SHOW': QC.SHOW,
    'SHOW_INTERVAL': 1.0 / 60,
}