# ATLAS
**Autonomous Target Location Acquisition System**

ATLAS is a customizable computer vision tool using YOLOv8, with PyTorch for model training and OpenCV for deployment. It detects targets and obstacles from a screen capture or webcam feed, and includes a full data pipeline for collecting, annotating, and training custom models.

---

## Pipeline

```
collect → review → assemble → train → run
```

| Step | Script | Description                                                                           |
|------|--------|---------------------------------------------------------------------------------------|
| 1 | `collect.py` | Captures screen frames and runs YOLOv8 to generate bounding box suggestions           |
| 2 | `review.py` | Manually annotate suggestions: mark as target, obstacle, or discard                   |
| 3 | `assemble.py` | Merges reviewed runs into a train/val dataset with a YOLO-compatible YAML             |
| 4 | `train.py` | Fine-tunes YOLOv8 on your dataset and exports to ONNX (and TensorRT if GPU available) |
| 5 | `atlas.py` | Runs the live detector on screen capture or webcam feed                               |

---

## Modes

- **Screen capture:** captures the full monitor and blanks the ATLAS window to avoid self-detection
- **Camera:**  reads from a webcam using the index set in config

```
atlas.py
atlas_camera.py
```

---

## Controls

**Reviewer**

| Key | Action |
|-----|--------|
| `D` (no selection) | Advance frame, drop unreviewed suggestions |
| `D` (box selected) | Mark as **target** |
| `E` (box selected) | Mark as **obstacle** |
| Left click | Select box / draw new box |
| Right click | Delete box or mark selected as obstacle |
| `Z` | Delete selected box |
| `A` | Previous frame |
| `Q` | Quit and save |

**Deployment:** hotkeys are set in `_config.yml`

---

## Configuration

All settings live in `user/_config.yml`:

```yaml
model:
  version: target_detector
  size: s                  # n, s, m, l, x (fastest/inaccurate → slowest/accurate)
  confidence: 0.5          # lower → more detections, more false positives
  iou: 0.3                 # lower → more duplicate suppression, fewer distinct detections
  classes:
    target: 0
    obstacle: 1

display:
  canvas_width: 600
  canvas_height: 420

screen:
  width: 1280
  height: 1024

camera:
  index: 0

hotkeys:
  toggle_active: alt+a
  quit: alt+q
```

---

## Classes

| ID | Label | Color |
|----|-------|-------|
| 0 | target | green |
| 1 | obstacle | red |

New labels are saved with class `-1` (unreviewed) and must be assigned during review before they are written to disk.

---

## Requirements

- Python 3.10+
- `ultralytics`, `torch`, `opencv-python`, `mss`, `keyboard`, `pyyaml`

```
pip install ultralytics torch opencv-python mss keyboard pyyaml
```

A CUDA-capable GPU is optional but recommended for training. CPU training is supported with reduced batch size.