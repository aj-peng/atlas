# DATASET TRAINING
from ultralytics import YOLO
from pathlib import Path
import torch
from datetime import datetime
import yaml

class Trainer:
    def __init__(self):
        self.ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.ROOT / "data"
        self.MODELS_DIR = self.DATA_DIR / "models"
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.CONFIG_PATH = self.ROOT / "user" / "_config.yml"

        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self.version = cfg['model']['version']
        self.size = cfg['model']['size']

        self.DATASET_DIR = self.DATA_DIR / "datasets" / self.version
        self.yaml_path = self.DATASET_DIR / "dataset.yaml"
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.yaml_path}")

    def train(self, epochs=100, imgsz=640):
        model_name = f"yolov8{self.size}_{self.version}"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training: {model_name}")

        model = YOLO(str(Path(__file__).parent / f'yolov8{self.size}.pt'))

        cuda_available = torch.cuda.is_available()
        device = 0 if cuda_available else 'cpu'
        print(f"Using device: {'CUDA' if cuda_available else 'CPU'}")

        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        results = model.train(
            data=str(self.yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            batch=16 if cuda_available else 4,
            workers=8 if cuda_available else 0,
            amp=False,
            name=model_name,
            project=str(self.MODELS_DIR),
            exist_ok=True,
            pretrained=True,
            resume=False,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=10,
            patience=300,
            save=True,
            save_period=10,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            nbs=64,
            val=True,
            plots=True,
        )

        if results is None:
            raise RuntimeError("Error: Invalid results")

        best_model_path = self.MODELS_DIR / model_name / "weights" / "best.pt"

        if best_model_path.exists():
            model = YOLO(best_model_path)

            try:
                model.export(format='onnx', imgsz=imgsz)
                print("ONNX export successful")
            except Exception as e:
                print(f"ONNX export error: {e}")

            if cuda_available:
                try:
                    model.export(format='engine', imgsz=imgsz, half=True)
                    print("TensorRT export successful")
                except Exception as e:
                    print(f"TensorRT export error: {e}")

            print(f"\nModel saved to: {best_model_path}, {self.MODELS_DIR}")
            return best_model_path
        else:
            raise FileNotFoundError(f"Best model not found at {best_model_path}")

if __name__ == '__main__':
    try:
        trainer = Trainer()
        trainer.train(epochs=100, imgsz=640)
    except KeyboardInterrupt:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupted")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
