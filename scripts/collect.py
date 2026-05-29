# DATA COLLECTION
from ultralytics import YOLO
import cv2
import mss
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml

class Collector:
    def __init__(self):
        self.ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT / "data"
        self.CONFIG_PATH = self.ROOT / "user" / "_config.yml"
        with open(self.CONFIG_PATH, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.RUN_DIR = self.DATA_DIR / "runs" / self.session

        # Create dataset structure
        (self.RUN_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.RUN_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)

        self.model = YOLO('yolov8n.pt')
        self.frame_count = 0

    def collect(self, fps=30, duration=30, confidence=0.5):
        """Collect frames directly instead of video"""
        print(f"[{self._timestamp()}] Starting collection: {duration}s at {fps} FPS")

        try:
            with mss.MSS() as sct:
                monitor = {
                    "top": 0,
                    "left": 0,
                    "width": self.cfg["screen"]["width"],
                    "height": self.cfg["screen"]["height"]
                }
                for i in range(duration * fps):
                    try:
                        screenshot = np.array(sct.grab(monitor))
                        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

                        results = self.model(frame, verbose=False)

                        if len(results[0].boxes) > 0:
                            frame_name = f"{self.session}_frame_{self.frame_count:06d}"

                            img_path = self.RUN_DIR / "images" / "train" / f"{frame_name}.jpg"
                            cv2.imwrite(str(img_path), frame)

                            lbl_path = self.RUN_DIR / "labels" / "train" / f"{frame_name}.txt"
                            saved = self._save_yolo_labels(results[0], frame.shape, str(lbl_path), confidence)

                            if saved:
                                self.frame_count += 1
                                print(f"[{self._timestamp()}] Collected: {self.frame_count} frames", end='\r')
                            else:
                                img_path.unlink(missing_ok=True)

                    except Exception as e:
                        print(f"\n[{self._timestamp()}] Frame {i} error: {e}")
                        continue

                    cv2.waitKey(int(1000 / fps))

        except KeyboardInterrupt:
            print(f"\n[{self._timestamp()}] Collected: {self.frame_count} frames")
        except mss.exception.ScreenShotError as e:
            print(f"\n[{self._timestamp()}] Error: {e}")
        except Exception as e:
            print(f"\n[{self._timestamp()}] Error: {e}")
        finally:
            print(f"[{self._timestamp()}] Saved: {self.RUN_DIR}")

    def _save_yolo_labels(self, result, frame_shape, save_path, confidence):
        """Save detections in YOLO format - force all to unreviewed (-1)"""
        h, w = frame_shape[:2]
        saved_any = False

        try:
            with open(save_path, 'w') as f:
                for box in result.boxes:
                    if float(box.conf[0]) > confidence:
                        cls = -1 # Force class ID to -1 (UNSET) instead of keeping COCO ID
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        saved_any = True
        except OSError as e:
            print(f"\n[{self._timestamp()}] Failed to save label {save_path}: {e}")

        return saved_any

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

if __name__ == '__main__':
    try:
        collector = Collector()
        collector.collect(fps=30, duration=1)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
