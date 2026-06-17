import random
import shutil
import yaml
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with open(root / "_config.yml", 'r') as cfg:
    config = yaml.safe_load(cfg)

layout = config['layout']
names = config['names']
seed = config['split']['seed']
val_ratio = config['split']['val']

dataset = Path(root / config['dataset']).resolve()
base = dataset / config['model']['version']

raw_images = base / layout['images'] / layout['raw']
raw_labels = base / layout['labels'] / layout['raw']

def pairs():
    """Frames that have a label file. Empty label = kept negative; missing = unlabelled, skipped."""
    result = []
    for img in sorted(raw_images.glob('frame_*.png')):
        label = raw_labels / f'{img.stem}.txt'
        if label.exists():
            result.append((img, label))
    return result

def reset(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def write_yaml():
    data_yaml = base / 'data.yaml'
    content = {
        'path':  base.as_posix(),
        'train': f"{layout['images']}/{layout['train']}",
        'val':   f"{layout['images']}/{layout['val']}",
        'names': names
    }
    with open(data_yaml, 'w') as f:
        yaml.safe_dump(content, f, sort_keys=False)
    print(f'Wrote {data_yaml}')

def build():
    data = pairs()
    if not data:
        print(f'No labelled frames in {raw_images} / {raw_labels}')
        return

    random.seed(seed)
    random.shuffle(data)
    n_val = int(len(data) * val_ratio)
    split = {layout['val']: data[:n_val], layout['train']: data[n_val:]}

    for name in (layout['train'], layout['val']):
        reset(base / layout['images'] / name)
        reset(base / layout['labels'] / name)

    for name, items in split.items():
        for img, label in items:
            shutil.copy(img, base / layout['images'] / name / img.name)
            shutil.copy(label, base / layout['labels'] / name / label.name)

    write_yaml()
    print(f'Split train: {len(split[layout["train"]])}  val: {len(split[layout["val"]])}')

if __name__ == "__main__":
    try:
        build()
    except KeyboardInterrupt:
        print(f'Error: {KeyboardInterrupt}')
    except Exception as err:
        print(f'Error: {err}')
