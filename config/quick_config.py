# YOLO模型
YOLO_MODEL      = 'test_nashe_640x640_nv12.hbm'

# 相机
FRONT_DEVICE    = '/dev/video0'
BOTTOM_DEVICE   = '/dev/video2'
FRONT_FPS       = 250
BOTTOM_FPS      = 250

# 识别目标
FRONT_TARGETS   = ['red-ball']
BOTTOM_TARGETS  = ['yellow-ball']
CLASS_NAMES     = ['door', 'red-ball', 'yellow-ball']

# 检测阈值
SCORE_THRES     = 0.6
NMS_THRES       = 0.45

# 功能
ENABLE_FRONT        = True  # 前视相机
ENABLE_BOTTOM       = True  # 下视相机
ENABLE_FRONT_YOLO   = True  # 前视YOLO
ENABLE_BOTTOM_YOLO  = True  # 下视YOLO

# 输出
SHOW            = False     # 桌面环境下显示
ENABLE_TIMING   = True      # 命令行环境下的统计信息
SIMPLE_TIMING   = True      # 命令行环境下的简化统计信息
ENABLE_LOG      = False     # 帧结果写 logs/