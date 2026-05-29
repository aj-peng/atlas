# DATA REVIEW
import cv2
from pathlib import Path
from datetime import datetime

CLASS_TARGET = 0
CLASS_OBSTACLE = 1
CLASS_UNSET = -1  # suggestion box not yet assigned by reviewer

class Reviewer:
    def __init__(self, run_dir: str | None = None):
        self.ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT / "data"

        if run_dir is None:
            runs = sorted((self.DATA_DIR / "runs").iterdir())
            if not runs:
                raise FileNotFoundError(f"No runs found in {self.DATA_DIR / 'runs'}")
            self.RUN_DIR = runs[-1]
        else:
            self.RUN_DIR = self.DATA_DIR / "runs" / run_dir

        self.IMG_DIR = self.RUN_DIR / "images" / "train"
        self.LBL_DIR = self.RUN_DIR / "labels" / "train"

        self.images = sorted(self.IMG_DIR.glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.IMG_DIR}")

        self.idx = 0
        self.boxes = []  # [cls, xc, yc, w, h]  cls=-1 means unreviewed suggestion
        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        self.selected = None
        self.img = None
        self.dirty = False

        self.reviewed = set()

        print(f"[{self._timestamp()}] Loaded {len(self.images)} frames from {self.RUN_DIR.name}")
        self._print_controls()

    def review(self):
        cv2.namedWindow("Reviewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reviewer", 1280, 720)
        cv2.setMouseCallback("Reviewer", self._mouse)

        while self.idx < len(self.images):
            self._load_frame()
            self._redraw()

            while True:
                key = cv2.waitKey(20) & 0xFF
                if cv2.getWindowProperty("Reviewer", cv2.WND_PROP_VISIBLE) < 1:
                    self._save_labels(force=True)
                    print(f"\n[{self._timestamp()}] EXIT: reviewed:{len(self.reviewed)}")
                    cv2.destroyAllWindows()
                    return

                # [Q] Save only reviewed boxes before quitting
                if key == ord('q'):
                    self._save_labels(force=True)
                    print(f"\n[{self._timestamp()}] EXIT: reviewed:{len(self.reviewed)}")
                    cv2.destroyAllWindows()
                    return

                # [D] mark selected box as TARGET; if nothing selected, advance frame
                elif key == ord('d'):
                    if self.selected is not None:
                        self.boxes[self.selected][0] = CLASS_TARGET
                        self.selected = None
                        self.dirty = True
                        self._redraw()
                    else:
                        self._save_labels(force=True)
                        if self.idx not in self.reviewed:
                            self.reviewed.add(self.idx)
                        self.idx += 1
                        break

                # [E] mark selected box as OBSTACLE
                elif key == ord('e'):
                    if self.selected is not None:
                        self.boxes[self.selected][0] = CLASS_OBSTACLE
                        self.selected = None
                        self.dirty = True
                        self._redraw()

                # [Z] delete selected box
                elif key == ord('z'):
                    if self.selected is not None:
                        self.boxes.pop(self.selected)
                        self.selected = None
                        self.dirty = True
                        self._redraw()

                # [A] previous frame
                elif key == ord('a'):
                    if self.idx > 0:
                        self._save_labels(force=True)
                        self.idx -= 1
                        break

                # [S] deselect
                elif key == ord('s'):
                    self.selected = None
                    self._redraw()

                if self.drawing and self.draw_current:
                    self._redraw()

        self.boxes = [b for b in self.boxes if b[0] != CLASS_UNSET]
        self._save_labels(force=True)
        print(f"\n[{self._timestamp()}] EXIT: reviewed:{len(self.reviewed)}")
        cv2.destroyAllWindows()

    def _load_frame(self):
        img_path = self.images[self.idx]
        lbl_path = self.LBL_DIR / img_path.with_suffix('.txt').name

        self.img = cv2.imread(str(img_path))
        self.boxes = []
        self.selected = None
        self.dirty = False

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    self.boxes.append([cls] + coords)

    def _save_labels(self, force=False):
        if (not force and not self.dirty) or self.idx >= len(self.images):
            return

        img_path = self.images[self.idx]
        lbl_path = self.LBL_DIR / img_path.with_suffix('.txt').name

        with open(lbl_path, 'w') as f:
            for box in self.boxes:
                cls, xc, yc, w, h = box
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        self.dirty = False

    def _redraw(self):
        if self.img is None:
            return

        display = self.img.copy()
        h, w = display.shape[:2]

        for i, box in enumerate(self.boxes):
            cls, xc, yc, bw, bh = box
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            if i == self.selected:
                color = (200, 200, 100)  # cyan: selected
            elif cls == CLASS_TARGET:
                color = (100, 200, 100)  # green: target
            elif cls == CLASS_OBSTACLE:
                color = (100, 100, 200)  # red: obstacle
            else:
                color = (200, 200, 200)  # white: unknown

            label = {CLASS_TARGET: "target", CLASS_OBSTACLE: "obstacle", CLASS_UNSET: "?"}.get(cls, f"cls:{cls}")
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.drawing and self.draw_start and self.draw_current:
            cv2.rectangle(display, self.draw_start, self.draw_current, (255, 100, 0), 2)

        total = len(self.images)
        unset = sum(1 for b in self.boxes if b[0] == CLASS_UNSET)
        assigned = sum(1 for b in self.boxes if b[0] != CLASS_UNSET)
        info = f"[{self.idx + 1}/{total}] suggestions:{unset} assigned:{assigned} reviewed:{len(self.reviewed)}"

        cv2.putText(display, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, self.images[self.idx].stem, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
                    cv2.LINE_AA)
        cv2.imshow("Reviewer", display)

    def _mouse(self, event, x, y, _flag, _param):
        if self.img is None:
            return
        h, w = self.img.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = self._box_at(x, y, w, h)
            if clicked is not None:
                if self.selected is not None:
                    self.boxes[self.selected][0] = CLASS_TARGET
                    self.selected = None
                    self.dirty = True
                    self._redraw()
                else:
                    self.selected = clicked
                    self.drawing = False

            else:
                self.selected = None
                self.drawing = True
                self.draw_start = (x, y)
                self.draw_current = (x, y)
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.draw_current = (x, y)
            self._redraw()

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.draw_start and abs(x - self.draw_start[0]) > 10 and abs(y - self.draw_start[1]) > 10:
                x1, y1 = self.draw_start
                x2, y2 = x, y
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                self.boxes.append([CLASS_TARGET, xc, yc, bw, bh])
                self.dirty = True
            self.draw_start = None
            self.draw_current = None
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked = self._box_at(x, y, w, h)
            if clicked is not None:
                if self.selected is not None:
                    self.boxes[self.selected][0] = CLASS_OBSTACLE
                    self.selected = None
                    self.dirty = True
                    self._redraw()

    def _box_at(self, x, y, w, h):
        for i, box in enumerate(self.boxes):
            cls, xc, yc, bw, bh = box
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return None

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def _print_controls(self):
        print("""
        Controls
        ────────────────────────────────────────────────────
        D (no selection)            advance frame, drop unreviewed suggestions
        D (box selected)            mark selected box as TARGET (green)
        E (box selected)            mark selected box as OBSTACLE (red)
        Left click                  select suggestion / draw new box / mark selected box as TARGET (green)
        Right click                 delete box under cursor / mark selected box as OBSTACLE (red)
        Z                           delete selected box
        S                           deselect
        A                           previous frame
        Q                           quit & save
        ────────────────────────────────────────────────────
        """)

if __name__ == '__main__':
    try:
        reviewer = Reviewer()
        reviewer.review()
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupted")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
