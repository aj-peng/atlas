import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
with open(root / "_config.yml", 'r') as cfg:
    config = yaml.safe_load(cfg)

theme = config['theme']
layout = config['layout']
predict_args = config['inference']
size = config['model']['size']
version = config['model']['version']

dataset = Path(root / config['dataset']).resolve()
raw_images = dataset / version / layout['images'] / layout['raw']
raw_labels = dataset / version / layout['labels'] / layout['raw']
pretrained = Path(root / config['pretrained']).resolve()

font = cv2.FONT_HERSHEY_SIMPLEX
hotkey = '[a]prev  [d]next  [w]keep  [s]del  [c]clear  [r]re-suggest  [esc]save+quit'
title = 'atlas // annotate'
tbar, bbar = 32, 32

class Annotate:
    def __init__(self):
        self.images = sorted(raw_images.glob('frame_*.png'))
        self.model = YOLO(pretrained / f'yolov8{size}.pt')
        self.names = self.model.names

        self.running = True
        self.boxes = []
        self.index = 0
        self.selected = -1
        self.drag_now = None
        self.drag_start = None

        self.path = None
        self.img = None
        self.w, self.h = 0, 0

    def run(self):
        if not self.images:
            print(f'Error: No images in {raw_images}')
            return
        raw_labels.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self.on_mouse)
        self.load_image(self.index)

        try:
            while self.running:
                display = self.draw_ui()
                cv2.imshow(title, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    self.save_current()
                    self.index = min(self.index + 1, len(self.images) - 1)
                    self.load_image(self.index)
                elif key == ord('a'):
                    self.save_current()
                    self.index = max(self.index - 1, 0)
                    self.load_image(self.index)
                elif key == ord('w') and self.selected >= 0:
                    self.boxes[self.selected][4] = -1
                    self.selected = -1
                elif key == ord('s') and self.selected >= 0:
                    del self.boxes[self.selected]
                    self.selected = -1
                elif key == ord('c'):
                    self.boxes.clear()
                elif key == ord('r'):
                    self.boxes = self.suggest(self.img)
                elif key == 27:
                    self.save_current()
                    self.running = False
                    break
        finally:
            cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            if abs(x - x0) > 5 and abs(y -y0) > 5:
                self.boxes.append([min(x0, x), min(y0, y), max(x0, x), max(y0, y), -1])
            else:
                self.selected = self.box_at(x, y)
            self.drag_start = None
            self.drag_now = None

    def load_image(self, i):
        self.path = self.images[i]
        self.img = cv2.imread(str(self.path))
        if self.img is None:
            raise FileNotFoundError(self.path)
        self.h, self.w = self.img.shape[:2]
        self.selected = -1

        label = raw_labels / f'{self.path.stem}.txt'
        if label.exists():
            self.boxes = self.load_labels(label)
        else:
            self.boxes = self.suggest(self.img)

    def load_labels(self, path: Path):
        boxes = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = line.split(' ')
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            x1 = int((cx - bw / 2) * self.w)
            y1 = int((cy - bh / 2) * self.h)
            x2 = int((cx + bw / 2) * self.w)
            y2 = int((cy + bh / 2) * self.h)
            boxes.append([x1, y1, x2, y2, -1])
        return boxes

    def to_label(self, box):
        x1, y1, x2, y2, _ = box
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        cx = ((xmin + xmax) / 2) / self.w
        cy = ((ymin + ymax) / 2) / self.h
        bw = (xmax - xmin) / self.w
        bh = (ymax - ymin) / self.h
        return f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}'

    def box_at(self, x, y):
        for idx in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _ = self.boxes[idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return -1

    def save_current(self):
        label = raw_labels / f'{self.path.stem}.txt'
        lines = [self.to_label(b) for b in self.boxes if b[4] == -1]
        label.write_text('\n'.join(lines))

    def suggest(self, img: np.ndarray):
        boxes = []
        result = self.model.predict(img, verbose=False, **predict_args)[0]
        for xyxy, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append([x1, y1, x2, y2, int(cls)])
        return boxes

    def draw_ui(self):
        display = self.img.copy()
        for idx, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
            if idx == self.selected:
                color = theme['color_5']
            elif cls == -1:
                color = theme['color_4']
            else:
                color = theme['color_3']
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, self.names.get(cls, 'keep'), (x1, y1 - 4), font, 0.5, color, 1)
        if self.drag_start and self.drag_now:
           cv2.rectangle(display, self.drag_start, self.drag_now, theme['color_6'], 1)

        cv2.rectangle(display, (0, 0), (self.w, tbar), theme['color_1'], -1)
        cv2.putText(display, title, (12, 22), font, 0.6, theme['color_6'], 2)
        cv2.putText(display, f'{self.path.name} [{self.index + 1}/{len(self.images)}]   boxes: {len(self.boxes)}',
                    (200, 22), font, 0.6, theme['color_2'], 1)
        cv2.rectangle(display, (0, self.h - bbar), (self.w, self.h), theme['color_1'], -1)
        cv2.putText(display, hotkey, (12, self.h - 10), font, 0.6, theme['color_2'], 1)

        return display

if __name__ == '__main__':
    try:
        Annotate().run()
    except KeyboardInterrupt:
        print(f'Error: {KeyboardInterrupt}')
    except Exception as err:
        print(f'Error: {err}')