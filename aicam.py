import os
import math
import sys
import threading
import time
from pathlib import Path

import libcamera
from PIL import Image, ImageDraw
from picamera2 import Picamera2

# VNC-friendly shim (package in project root: displayhatmini/__init__.py)
from displayhatmini import DisplayHATMini

# PyCoral / TFLite
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from tflite_runtime.interpreter import Interpreter

# -----------------------------
# Filesystem / paths
# -----------------------------
script_path = Path(__file__).resolve()
script_dir = script_path.parent                        # .../PiZeroAiCam/src
project_root = (script_dir / "..").resolve()           # .../PiZeroAiCam

# -----------------------------
# Model + interpreter (EdgeTPU -> CPU fallback)
# -----------------------------
edgetpu_model = str(project_root / "model/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
cpu_model     = str(project_root / "model/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess.tflite")

try:
    interpreter = make_interpreter(edgetpu_model)
    print("Using Edge TPU delegate")
except Exception as e:
    print(f"Edge TPU unavailable ({e}); falling back to CPU TFLite.")
    model_path = cpu_model if os.path.exists(cpu_model) else edgetpu_model
    interpreter = Interpreter(model_path)

interpreter.allocate_tensors()

# -----------------------------
# Display
# -----------------------------
width  = DisplayHATMini.WIDTH
height = DisplayHATMini.HEIGHT
disp_buffer = Image.new("RGB", (width, height))
displayhatmini = DisplayHATMini(disp_buffer)

# -----------------------------
# Camera
# -----------------------------
picam2 = Picamera2()
capture_config = picam2.create_still_configuration(
    main={"size": (width, height), "format": "RGB888"},
    lores=None,
    raw=None,
    colour_space=libcamera.ColorSpace.Raw(),
    buffer_count=6,
    controls={"AfMode": libcamera.controls.AfModeEnum.Continuous},
    queue=True
)
picam2.configure(capture_config)
picam2.start()

# -----------------------------
# Labels
# -----------------------------
labels = read_label_file(str(project_root / "model/mobilenet_coco/coco_labels.txt"))

# -----------------------------
# Inference thread + helpers
# -----------------------------
detected_objs = []
inference_latency = sys.float_info.max
image_buffer = Image.new("RGB", (width, height))

def is_duplicate(center1, center2, dist_thresh=15):
    return dist_thresh >= math.dist(center1, center2)

def run_interpreter():
    global image_buffer, detected_objs, inference_latency
    start = time.perf_counter()

    # Resize/copy current frame into input tensor
    _, scale = common.set_resized_input(
        interpreter,
        image_buffer.size,
        lambda size: image_buffer.resize(size, Image.ANTIALIAS)
    )
    interpreter.invoke()
    inference_latency = time.perf_counter() - start

    # Get and lightly de-duplicate detections
    objs = detect.get_objects(interpreter, 0.4, scale)
    dedup_map = {}
    filtered = []
    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2, (bbox.ymax + bbox.ymin) / 2)
        bucket = dedup_map.get(obj.id)
        if bucket is not None and any(is_duplicate(center, c) for c in bucket):
            continue
        dedup_map.setdefault(obj.id, []).append(center)
        filtered.append((obj, bbox))
    detected_objs[:] = filtered

inference_thread = threading.Thread(target=run_interpreter, daemon=True)
last_frame_time = time.perf_counter()
framerate = 0.0

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        # Capture current camera frame
        frame = picam2.capture_image()             # PIL Image (RGB)
        image_buffer = frame.copy()                # save for the inference thread

        # Kick off inference if the previous one finished
        if not inference_thread.is_alive():
            inference_thread = threading.Thread(target=run_interpreter, daemon=True)
            inference_thread.start()

        # Draw overlays on the frame
        draw = ImageDraw.Draw(frame)
        for obj, bbox in detected_objs:
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="yellow")
            draw.text(
                (bbox.xmin + 10, bbox.ymin + 10),
                f"{labels.get(obj.id, obj.id)}\n{obj.score:.2f}",
                fill="yellow"
            )
        draw.text((10, 10), f"{int(framerate):02d} fps\n{inference_latency * 1000:.2f} ms", fill="white")

        # Match original orientation if needed
        frame_to_show = frame.transpose(Image.ROTATE_180)

        # Show on the VNC window (shim expects a PIL image)
        displayhatmini.display(frame_to_show)

        # FPS calc
        this_frame_time = time.perf_counter()
        framerate = 1.0 / max(this_frame_time - last_frame_time, 1e-6)
        last_frame_time = this_frame_time

finally:
    if inference_thread.is_alive():
        inference_thread.join()
