import torch
import yaml
from pathlib import Path
from ultralytics import YOLO, settings

settings.update({'mlflow': False}) # disable MLflow integration (file-store crash)

root = Path(__file__).resolve().parents[1]
with open(root / "_config.yml", 'r') as cfg:
    config = yaml.safe_load(cfg)

tcfg = config['train']
aug = config['augment']
size = config['model']['size']
version = config['model']['version']

dataset = Path(root / config['dataset']).resolve()
data_yaml = dataset / version / 'data.yaml'
pretrained = Path(root / config['pretrained']).resolve()
trained = Path(root / config['trained']).resolve()

def resolve_device():
    if torch.cuda.is_available():
        print(f'Training on GPU: {torch.cuda.get_device_name(0)}')
        return 0
    elif torch.backends.mps.is_available():
        print('Training on GPU: Apple Silicon (MPS)')
        return 'mps'
    print('WARNING: no GPU available: Training on CPU will be very slow.')
    return 'cpu'

def train(augment=False):
    device = resolve_device()

    model = YOLO(pretrained / f'yolov8{size}.pt')
    model.train(
        data=str(data_yaml),
        epochs=tcfg['epochs'],
        imgsz=tcfg['imgsz'],
        batch=tcfg['batch'],
        patience=tcfg['patience'],
        workers=0,
        device=device,
        project=str(trained),
        name=version,
        exist_ok=True,
        **(aug if augment else {}),
    )

    save_dir = Path(model.trainer.save_dir)
    pictures = save_dir / 'graphs'
    pictures.mkdir(parents=True, exist_ok=True)
    for img in [*save_dir.glob('*.png'), *save_dir.glob('*.jpg')]:
        img.replace(pictures / img.name)

if __name__ == "__main__":
    try:
        train(augment=False)
    except KeyboardInterrupt:
        print(f'Error: {KeyboardInterrupt}')
    except Exception as err:
        print(f'Error: {err}')
