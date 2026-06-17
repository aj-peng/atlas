import cv2
import mss
import numpy as np
import threading
import time
import yaml
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
with open(root / '_config.yml', 'r') as cfg:
    config = yaml.safe_load(cfg)

theme = config['theme']
hotkey = config['hotkey']
predict_args = config['inference']
version = config['model']['version']
use_custom = config['model']['custom']

trained = Path(root / config['trained']).resolve()
custom_weights = trained / version / config['weights']

pretrained = Path(root / config['pretrained']).resolve()
v8base_weights = pretrained / f"yolov8{config['model']['size']}.pt"

region = {'top': 0, 'left': 0, 'width': config['capture']['width'], 'height': config['capture']['height']}
fps, pwidth, pheight = config['panel']['fps'], config['panel']['width'], config['panel']['height']

font = cv2.FONT_HERSHEY_SIMPLEX
title = f"{config['title']} // detector"
tbar, bbar = 48, 88
viewport = pheight - tbar - bbar

class ScreenCapture:
    def __init__(self):
        weights = custom_weights if use_custom else v8base_weights

        self.sct = mss.MSS()
        self.model = YOLO(model=weights)
        self.names = self.model.names
        self.start = time.time()
        self.running = True
        self.active = False
        self.targets = 0
        self.boxes = []
        self.frame = None

    def run(self):
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        threading.Thread(target=self.infer_loop, daemon=True).start()

        try:
            while self.running:
                frame = self.grab()
                self.frame = frame

                canvas = frame.copy()
                if self.active:
                    for x1, y1, x2, y2, cls, conf in self.boxes:
                        label = f"{self.names.get(cls, '')} {conf:.2f}"
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), theme['color_4'], 2)
                        cv2.putText(canvas, label, (x1, y1 - 4), font, 1, theme['color_4'], 2)

                display = self.draw_ui(canvas)
                cv2.imshow(title, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(hotkey['quit']):
                    self.quit()
                    break
                elif key == ord(hotkey['toggle']):
                    self.toggle()
        finally:
            cv2.destroyAllWindows()

    def infer_loop(self):
        while self.running:
            time.sleep(0.01)
            if not self.active:
                self.boxes = []
                self.targets = 0
                continue
            frame = self.frame
            if frame is None:
                continue
            result = self.model.predict(frame, verbose=False, **predict_args)[0]
            self.boxes = [
                [*map(int, xyxy), int(cls), float(conf)]
                for xyxy, cls, conf in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                )
            ]
            self.targets = len(self.boxes)

    def toggle(self):
        self.active = not self.active

    def quit(self):
        self.running = False

    def grab(self):
        capture = self.sct.grab(region)
        frame = np.array(capture)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def draw_ui(self, frame: np.ndarray):
        preview = cv2.resize(frame, (pwidth, viewport))
        display = np.zeros((pheight, pwidth, 3), dtype=np.uint8)
        display[tbar:tbar + viewport, 0:pwidth] = preview

        cv2.rectangle(display, (0, 0), (pwidth, tbar), theme['color_1'], -1)
        cv2.rectangle(display, (0, pheight - bbar), (pwidth, pheight), theme['color_1'], -1)

        cv2.putText(display,title,(16, 32),font,0.7,theme['color_6'],2)

        status_color = theme['color_4'] if self.active else theme['color_3']
        mins, secs = divmod(int(time.time() - self.start), 60)
        cv2.putText(display,f'active: {self.active}',(16, pheight - 64),font,0.5,status_color,1)
        cv2.putText(display,f'targets: {self.targets}',(16, pheight - 40),font,0.5,theme['color_2'],1)
        cv2.putText(display,f'session: {mins:02d}:{secs:02d}',(16, pheight - 16),font,0.5,theme['color_2'],1)

        cv2.putText(display, f"[{hotkey['toggle']}]: start/stop",(pwidth - 240, pheight - 40),font,0.5, theme['color_2'],1)
        cv2.putText(display, f"[{hotkey['quit']}]: quit",(pwidth - 240, pheight - 16),font,0.5, theme['color_2'],1)

        return display

if __name__ == "__main__":
    try:
        ScreenCapture().run()
    except KeyboardInterrupt:
        print(f'Error: {KeyboardInterrupt}')
    except Exception as err:
        print(f'Error: {err}')
