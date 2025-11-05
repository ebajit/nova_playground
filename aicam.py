import os
import math
import sys
import threading
import time
from pathlib import Path

import socket
import io
import struct
import json
import libcamera
from PIL import Image, ImageDraw
from picamera2 import Picamera2

# VNC-friendly shim (package in project root: displayhatmini/__init__.py)
from displayhatmini import DisplayHATMini
UDP_STREAM_PORT = 5000
_udp_target = None       # (ip, port)
_udp_sock = None
_udp_frame_id = 0
_udp_chunk_size = 1200   # bytes per UDP packet payload

try:
    import pi5_server
except Exception:
    pi5_server = None


def _handle_stream_command(msg, sender):
    """Handle connect/disconnect control messages from registered clients.

    Expected JSON: {"cmd":"connect","ip":"<ip optional>","port":<port optional>} or {"cmd":"disconnect"}
    If ip is not provided, the handler will try to resolve sender IP from pi5_server.user_database.
    """
    global _udp_target, _udp_sock
    if not isinstance(msg, dict):
        return
    cmd = msg.get('cmd')
    if cmd == 'connect':
        ip = msg.get('ip')
        port = int(msg.get('port', UDP_STREAM_PORT))
        if not ip and pi5_server and sender in getattr(pi5_server, 'user_database', {}):
            entry = pi5_server.user_database.get(sender)
            if entry and len(entry) >= 3:
                ip = entry[2][0]
        if not ip:
            try:
                if pi5_server:
                    pi5_server.dm_msg({"status": "error", "reason": "no_target_ip"}, sender)
            except Exception:
                pass
            return

        # create UDP socket if needed
        try:
            if _udp_sock is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # reduce internal buffering to keep latency low
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 20000)
                except Exception:
                    pass
                _udp_sock = s
            _udp_target = (ip, port)
            try:
                if pi5_server:
                    pi5_server.dm_msg({"status": "ok", "action": "udp_stream_set", "ip": ip, "port": port}, sender)
            except Exception:
                pass
        except Exception as e:
            try:
                if pi5_server:
                    pi5_server.dm_msg({"status": "error", "reason": str(e)}, sender)
            except Exception:
                pass

    elif cmd == 'disconnect':
        try:
            if _udp_sock:
                try:
                    _udp_sock.close()
                except Exception:
                    pass
            _udp_sock = None
            _udp_target = None
            if pi5_server:
                try:
                    pi5_server.dm_msg({"status": "ok", "action": "udp_stream_stopped"}, sender)
                except Exception:
                    pass
        except Exception:
            pass


# register handler with pi5_server if available
if pi5_server is not None:
    try:
        pi5_server.register_command_handler(_handle_stream_command)
    except Exception:
        pass

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

def pixels_to_mm(x_pixel, y_pixel, img_width, img_height, sensor_width_mm, sensor_height_mm):
    """
    Convert pixel coordinates to millimeters based on sensor size and image resolution.

    Args:
        x_pixel (int): X coordinate in pixels.
        y_pixel (int): Y coordinate in pixels.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.
        sensor_width_mm (float): Sensor width in millimeters.
        sensor_height_mm (float): Sensor height in millimeters.

    Returns:
        (float, float): (x_mm, y_mm) coordinates in millimeters.
    """
    x_mm = (x_pixel / img_width) * sensor_width_mm
    y_mm = (y_pixel / img_height) * sensor_height_mm
    return x_mm, y_mm


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

# Arducam Model 3 (IMX477) sensor specs
SENSOR_WIDTH_MM = 6.287
SENSOR_HEIGHT_MM = 4.712
SENSOR_RESOLUTION = (4056, 3040)  # (width, height)

# Example usage:
# Convert pixel (x_pixel, y_pixel) to mm
# x_mm, y_mm = pixels_to_mm(x_pixel, y_pixel, SENSOR_RESOLUTION[0], SENSOR_RESOLUTION[1], SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM)

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
            # Calculate center in pixels
            center_x = (bbox.xmax + bbox.xmin) / 2
            center_y = (bbox.ymax + bbox.ymin) / 2

            # Convert center from pixels to millimeters
            center_mm = pixels_to_mm(
                center_x, center_y,
                SENSOR_RESOLUTION[0], SENSOR_RESOLUTION[1],
                SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
            )
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="yellow")
            draw.text(
                (bbox.xmin + 10, bbox.ymin + 10),
                f"{labels.get(obj.id, obj.id)}\n{obj.score:.2f}",
                fill="yellow"
            )
        draw.text((10, 10), f"{int(framerate):02d} fps\n{inference_latency * 1000:.2f} ms", fill="white")

        # Match original orientation if needed
        frame_to_show = frame.transpose(Image.ROTATE_180)

        # Encode JPEG once and send over UDP to target if requested
        try:
            buf = io.BytesIO()
            frame_to_show.save(buf, format='JPEG', quality=80)
            jpeg_bytes = buf.getvalue()
            # send segmented over UDP
            if _udp_target and _udp_sock:
                try:
                    # determine chunking
                    chunk_sz = _udp_chunk_size
                    total = (len(jpeg_bytes) + chunk_sz - 1) // chunk_sz
                    _udp_frame_id = (_udp_frame_id + 1) & 0xFFFFFFFF
                    fid = _udp_frame_id
                    for idx in range(total):
                        start = idx * chunk_sz
                        chunk = jpeg_bytes[start:start+chunk_sz]
                        hdr = struct.pack('>IHH', fid, total, idx)
                        _udp_sock.sendto(hdr + chunk, _udp_target)
                except Exception:
                    try:
                        _udp_sock.close()
                    except Exception:
                        pass
                    _udp_sock = None
                    _udp_target = None
        except Exception:
            pass

        # Show on the VNC window (shim expects a PIL image)
        displayhatmini.display(frame_to_show)

        # FPS calc
        this_frame_time = time.perf_counter()
        framerate = 1.0 / max(this_frame_time - last_frame_time, 1e-6)
        last_frame_time = this_frame_time

finally:
    if inference_thread.is_alive():
        inference_thread.join()
