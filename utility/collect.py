import cv2
import mss
import numpy as np
import time
import yaml
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with open(root / "_config.yml", 'r') as cfg:
    config = yaml.safe_load(cfg)

theme = config['theme']
layout = config['layout']
hotkey = config['hotkey']
target = config['target']

dataset = Path(root / config['dataset']).resolve()
save_dir = dataset / config['model']['version'] / layout['images'] / layout['raw']

region = {'top': 0, 'left': 0, 'width': config['capture']['width'], 'height': config['capture']['height']}
fps, pwidth, pheight = config['panel']['fps'], config['panel']['width'], config['panel']['height']

font = cv2.FONT_HERSHEY_SIMPLEX
title = f"{config['title']} // collect"
tbar, bbar = 48, 88
viewport = pheight - tbar - bbar

class Collect:
    def __init__(self):
        self.sct = mss.MSS()
        self.start = time.time()

        self.running = True
        self.active = False

        self.saved = len(list(save_dir.glob('frame_*.png')))
        self.last_save = 0.0

    def run(self):
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                frame = self.grab()
                if self.active:
                    self.save_frame(frame)

                display = self.draw_ui(frame)
                cv2.imshow(title, display)
                # Controls are handled by the focused panel window (binds from _config.yml).
                key = cv2.waitKey(1) & 0xFF
                if key == ord(hotkey['quit']):
                    self.quit()
                    break
                elif key == ord(hotkey['toggle']):
                    self.toggle()
                elif key == ord(hotkey['clear']):
                    self.clear()
        finally:
            cv2.destroyAllWindows()
            print(f'Saved {self.saved} frames to {save_dir}')

    def toggle(self):
        self.active = not self.active

    def quit(self):
        self.running = False

    def clear(self):
        self.active = False
        removed = 0
        for f in list(save_dir.glob('frame_*.png')):
            f.unlink()
            removed += 1
        self.saved = 0
        print(f'Cleared {removed} frames from {save_dir}')

    def grab(self):
        capture = self.sct.grab(region)
        frame = np.array(capture)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def save_frame(self, frame: np.ndarray):
        if self.saved >= target:
            self.active = False
            return

        now = time.time()
        if (now - self.last_save) < 1.0 / fps:
            return
        self.last_save = now

        path = save_dir / f'frame_{self.saved:06d}.png'
        cv2.imwrite(str(path), frame)
        self.saved += 1

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
        cv2.putText(display,f'saved: {self.saved} / {target}',(16, pheight - 40),font,0.5,theme['color_2'],1)
        cv2.putText(display,f'session: {mins:02d}:{secs:02d}',(16, pheight - 16),font,0.5,theme['color_2'],1)

        cv2.putText(display, f"[{hotkey['toggle']}]: start/stop",(pwidth - 240, pheight - 64),font,0.5, theme['color_2'],1)
        cv2.putText(display, f"[{hotkey['clear']}]: clear",(pwidth - 240, pheight - 40),font,0.5, theme['color_2'],1)
        cv2.putText(display, f"[{hotkey['quit']}]: quit",(pwidth - 240, pheight - 16),font,0.5, theme['color_2'],1)

        return display

if __name__ == '__main__':
    try:
        Collect().run()
    except KeyboardInterrupt:
        print(f'Error: {KeyboardInterrupt}')
    except Exception as err:
        print(f'Error: {err}')
