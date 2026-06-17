## ATLAS

A computer vision tool built with YOLOv8 for live detection of targets (screen or camera).
It covers the full process: capture frames > annotate > build a dataset > train/fine-tune a
model > run live detection. The workflow is driven by `_config.yml`.

## Pipeline

| Script                | Description                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------|
| `utility/collect.py`  | Captures screen frames into `.dataset/<version>/images/raw`.                               |
| `utility/annotate.py` | Draw/adjust boxes per frame (YOLOv8 pre-suggests), writes YOLO `.txt` labels.              |
| `scripts/build.py`    | Splits labelled frames into train/val and writes `data.yaml`.                              |
| `scripts/train.py`    | Fine-tunes `yolov8<size>.pt` on the dataset; weights land in `.models/trained/<version>/`. |
| `detector/`           | Loads the trained (or base) weights and runs live inference.                               |

Typical run order:

```bash
python utility/collect.py     # record frames
python utility/annotate.py    # label them
python scripts/build.py       # train/val split + data.yaml
python scripts/train.py       # fine-tune
python detector/screen.py     # run it
```

## Configuration (`_config.yml`)

One file controls every script. Key sections:

- **`model`**:`version` (dataset/run name, e.g. `v1`), `size` (`n`/`s`/`m`/...), and
  `custom` (`True` = use your trained weights, `False` = base `yolov8<size>.pt`).
- **`capture`**: screen region `width`/`height` and webcam `camera` index.
- **`hotkey`**: panel controls: `toggle` (`e`), `clear` (`w`), `quit` (`q`).
- **`split`**: `val` ratio, `seed`, and `block` (frames kept together across the
  split so near-identical consecutive frames don't leak into val; set `block: 1`
  for datasets of unique, unrelated images).
- **`inference`**: `conf` (confidence threshold) and `iou` (NMS overlap).
- **`train`** / **`augment`**: epochs, image size, batch, patience, and augmentation
  strengths passed straight to Ultralytics.
- **`dataset` / `trained` / `pretrained` / `weights`**: paths for data and model files.

## Example: running the detector

```bash
# 1. Point the config at your trained model
#    _config.yml:
#      model:
#        version: v1
#        custom:  True        # load .models/trained/v1/weights/best.pt
#      inference:
#        conf: 0.5            # see note below

# 2. Launch the screen detector
python detector/screen.py
```

Controls (the panel window must be focused): **`e`** toggle detection on/off,
**`q`** quit. `detector/camera.py` is the same but reads a webcam instead of the screen.

> **Note on `conf`:** a freshly trained model on a small dataset is often
> under-confident. Check `BoxF1_curve.png` in `.models/trained/<version>/graphs/`
> for the confidence where F1 peaks, and set `inference.conf` near that value.
> If the detector shows nothing, lower `conf` (e.g. `0.05`) before assuming the
> model is broken.

## Requirements

Python **3.10+**. Install dependencies:

```bash
pip install ultralytics opencv-python mss numpy pyyaml
```

- `ultralytics` YOLOv8 (training + inference); pulls in `torch`.
- `opencv-python` capture, drawing, and the panel UI.
- `mss` fast screen grabbing.
- `numpy`, `pyyaml` array handling and config parsing.

For GPU training, install a CUDA build of `torch` (Apple Silicon uses MPS
automatically). `scripts/train.py` prints which device it selected on startup.
