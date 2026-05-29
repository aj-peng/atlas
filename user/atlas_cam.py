# ATLAS: Camera
from ultralytics import YOLO
import cv2
import numpy as np
import keyboard
import yaml
from pathlib import Path
from queue import Queue, Empty

# Load configuration
with open("_config.yml") as f:
    cfg = yaml.safe_load(f)

version = cfg["model"]["version"]
model_path = Path(__file__).parent.parent / "data" / "models" / f"yolov8n_{version}" / "weights" / "best.pt"

if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found at {model_path}\n")

print(f"Loading model: {model_path}")
MODEL = YOLO(str(model_path))
print(f"Model classes: {MODEL.names}")

ATLAS = "ATLAS"

class Atlas:
    def __init__(self):
        self.width = cfg["display"]["canvas_width"]
        self.height = cfg["display"]["canvas_height"]
        self.camera_index = cfg["camera"]["index"]

        self.cap = None
        self.active = False

        # Stats
        self.target_count = 0
        self.obstacle_count = 0

        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2.namedWindow(ATLAS, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(ATLAS, self.width, self.height)

        self.command_queue = Queue()

    def setup_global_hotkeys(self):
        keyboard.add_hotkey(cfg["hotkeys"]["toggle_active"], self.toggle_active)
        keyboard.add_hotkey(cfg["hotkeys"]["quit"], self.quit_program)

    def toggle_active(self):
        self.command_queue.put(('toggle_active',))

    def quit_program(self):
        self.command_queue.put(('quit',))

    def process_commands(self):
        try:
            while True:
                command = self.command_queue.get_nowait()
                if command[0] == 'toggle_active':
                    self.active = not self.active
                elif command[0] == 'quit':
                    return False
        except Empty:
            pass
        return True

    def draw_ui(self, frame=None):
        if frame is None:
            display = self.canvas.copy()
        else:
            display = cv2.resize(frame, (self.width, self.height))

        # Draw UI overlay
        cv2.rectangle(display, (0, 0), (self.width, 48), (30, 30, 30), -1)
        cv2.rectangle(display, (0, self.height - 88), (self.width, self.height), (30, 30, 30), -1)

        cv2.putText(display, ATLAS, (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 200), 2)

        status_color = (100, 200, 100) if self.active else (100, 100, 200)
        cv2.putText(display, f"ACTIVE: {self.active}", (16, self.height - 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        cv2.putText(display, f"TARGETS: {self.target_count}", (16, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"OBSTACLES: {self.obstacle_count}", (16, self.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(display, f"[{cfg['hotkeys']['toggle_active']}]: Start/Stop", (self.width - 240, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"[{cfg['hotkeys']['quit']}]: Quit", (self.width - 240, self.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    def capture_camera(self):
        """Read a frame directly from the webcam."""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def process_frame(self, frame):
        if not self.active or MODEL is None:
            return frame

        try:
            # We no longer need to blank the window, just pass the frame to YOLO
            inference_frame = frame.copy()

            results = MODEL(
                inference_frame,
                conf=cfg["model"]["confidence"],
                iou=cfg["model"].get("iou", 0.4),
                verbose=False
            )

            self.target_count = 0
            self.obstacle_count = 0
            annotated = frame.copy()

            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if cls == cfg["model"]["classes"]["target"]:
                        self.target_count += 1
                        color = (0, 255, 0)
                        label = f"target {conf:.2f}"
                    elif cls == cfg["model"]["classes"]["obstacle"]:
                        self.obstacle_count += 1
                        color = (0, 0, 255)
                        label = f"obstacle {conf:.2f}"
                    else:
                        continue  # Unknown class: do not render

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return annotated

        except Exception as de:
            print(f"ATLAS Detection Error: {de}")
            return frame

    def run(self):
        print("ATLAS: Running (Camera Mode)")

        # Initialize the webcam using the index from config (usually 0)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return

        self.setup_global_hotkeys()

        while True:
            if not self.process_commands():
                break

            if self.active:
                frame = self.capture_camera()
                if frame is not None:
                    frame = self.process_frame(frame)
                    display = self.draw_ui(frame)
                else:
                    display = self.draw_ui()
            else:
                display = self.draw_ui()

            cv2.imshow(ATLAS, display)
            cv2.waitKey(1)

        # Cleanup
        if self.cap is not None:
            self.cap.release()
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        print("ATLAS: Shutdown")


if __name__ == '__main__':
    try:
        atlas = Atlas()
        atlas.run()
    except Exception as e:
        print(f"ATLAS Error: {e}")
    except KeyboardInterrupt:
        print(f"ATLAS User Interrupt")
