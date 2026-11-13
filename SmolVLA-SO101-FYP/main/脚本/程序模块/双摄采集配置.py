"""双摄统一配置：桌面与夹爪均使用 640×480，保存及模型控制节拍统一为 20 Hz。"""
RECORD_FPS = 20
CAMERAS = {
    "front": {"feature": "observation.images.front", "index": 1, "fourcc": "MJPG",
              "input_fps": 25, "width": 640, "height": 480,
              "device": "1080P USB Camera", "label": "桌面相机"},
    "wrist": {"feature": "observation.images.wrist", "index": 2, "fourcc": "YUY2",
              "input_fps": 20, "width": 640, "height": 480,
              "device": "icspring camera", "label": "夹爪相机"},
}
# 桌面 640×480 MJPEG 驱动声明 25–30 FPS，录制线程按 20 Hz 保存。
# 录制线程按统一 20 Hz 从两路最新图像采样，不提高机械臂控制频率。
DEVICE_PROFILES = {spec["device"]: spec for spec in CAMERAS.values()}
