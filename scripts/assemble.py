# DATASET ASSEMBLE
import yaml
import shutil
from pathlib import Path
from datetime import datetime

class Assembler:
    def __init__(self):
        self.ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT / "data"
        self.RUNS_DIR = self.DATA_DIR / "runs"
        self.CONFIG_PATH = self.ROOT / "user" / "_config.yml"

        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self.DATASET_DIR = self.DATA_DIR / "datasets" / cfg['model']['version']

        self.CLASS_NAMES = {
            0: "target",
            1: "obstacle",
        }

    def build_dataset(self, val_split=0.2):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building dataset")

        for split in ['train', 'val']:
            (self.DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

        all_frames = []
        for run_dir in sorted(self.RUNS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue

            img_dir = run_dir / "images" / "train"
            lbl_dir = run_dir / "labels" / "train"

            if not img_dir.exists() or not lbl_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                reviewed_lines = [
                    line for line in lbl_path.read_text().splitlines()
                    if line.strip() and int(line.split()[0]) != -1
                ]
                if reviewed_lines:
                    all_frames.append((img_path, lbl_path))

        if not all_frames:
            raise ValueError("No reviewed frames found")

        import random
        random.seed(67)
        random.shuffle(all_frames)

        split_idx = int(len(all_frames) * (1 - val_split))
        train_frames = all_frames[:split_idx]
        val_frames = all_frames[split_idx:]

        print(f"Copying {len(train_frames)} train + {len(val_frames)} val frames")

        for frames, split in [(train_frames, 'train'), (val_frames, 'val')]:
            for i, (img_path, lbl_path) in enumerate(frames):
                new_name = f"{split}_{i:06d}"
                dst_img = self.DATASET_DIR / 'images' / split / f"{new_name}.jpg"
                dst_lbl = self.DATASET_DIR / 'labels' / split / f"{new_name}.txt"

                shutil.copy2(img_path, dst_img)
                reviewed_lines = [
                    line for line in lbl_path.read_text().splitlines()
                    if line.strip() and int(line.split()[0]) != -1
                ]
                dst_lbl.write_text("\n".join(reviewed_lines) + "\n" if reviewed_lines else "")

                if i % 100 == 0:
                    print(f"  {split}: {i}/{len(frames)}", end='\r')

        self._create_yaml()
        self._print_stats()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset ready: {self.DATASET_DIR}")
        return self.DATASET_DIR

    def _create_yaml(self):
        data_yaml = {
            'path': str(self.DATASET_DIR.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': self.CLASS_NAMES,
            'nc': len(self.CLASS_NAMES)
        }
        with open(self.DATASET_DIR / 'dataset.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    def _print_stats(self):
        """Count instances from the assembled dataset labels (post -1 strip)."""
        from collections import Counter
        class_counts = Counter()
        train_count = val_count = 0

        for split in ["train", "val"]:
            lbl_dir = self.DATASET_DIR / "labels" / split
            for lbl_path in sorted(lbl_dir.glob("*.txt")):
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        class_counts[int(parts[0])] += 1

        print(f"Total frames: {train_count + val_count}")
        print(f"    Train: {train_count}")
        print(f"    Val: {val_count}")
        print("\nInstances per class (assembled):")
        for cls_id, count in sorted(class_counts.items()):
            name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    {name}: {count}")

if __name__ == '__main__':
    try:
        assembler = Assembler()
        assembler.build_dataset(val_split=0.2)
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupted")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
