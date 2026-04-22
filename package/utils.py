import os
from pathlib import Path
import torch
from torchvision import transforms, datasets
from PIL import Image
import logging
from package.config import LOG_DIR
from package.pipeline import run_pipeline


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        data = datasets.ImageFolder(root=data_dir)
        self.samples = data.samples
        self.targets = data.targets
        self.classes = data.classes

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def ensure_pipeline_ready(
    disease_id: str,
    data_dir: str,
    layout: str,
    registry_dir: str,
    shards_dir: str,
    nfolds: int,
    target_size: int = 224,
    csv_path=None,
    auto_run: bool = True,
) -> bool:
    """
    Check whether the BDA registry exists for the given disease.
    If it doesn't exist and auto_run=True, run the pipeline to create it.

    Returns True if the registry is ready, False otherwise.
    """

    parquet_path = Path(registry_dir) / f"{disease_id}.parquet"

    if parquet_path.exists():
        return True

    if not auto_run:
        print(
            f"[utils] BDA registry not found: {parquet_path}\n"
            f"  Run the data pipeline first:\n"
            f"    python -m data_pipeline.pipeline --disease {disease_id} "
            f"--source_dir {data_dir} --layout {layout}"
        )
        return False

    print(f"[utils] Registry not found for '{disease_id}'. Running pipeline automatically...")
    try:
        run_pipeline(
            disease=disease_id,
            source_dir=data_dir,
            layout=layout,
            source="local",
            nfolds=nfolds,
            registry_dir=registry_dir,
            shards_dir=shards_dir,
            target_size=target_size,
            csv=csv_path,
            build_shards=True,
            compute_stats=True,
            validate=True,
            skip_if_exists=True,
        )
        return parquet_path.exists()
    except Exception as e:
        print(f"[utils] Pipeline auto-run failed: {e}")
        print("[utils] Falling back to legacy FullDataset.")
        return False


class Logger:
    def __init__(self, name: str = "Logger", file_name: str = "batch"):
        self.logger = logging.getLogger(name)
        self.current_log_dir = LOG_DIR / name
        self.current_log_dir.mkdir(exist_ok=True)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            console_formatter = logging.Formatter("[%(name)s] %(message)s")

            base_file_name = self.current_log_dir / file_name

            info_path = f"{str(base_file_name)}_debug.log"
            info_handler = logging.FileHandler(info_path, encoding="utf-8")
            info_handler.setLevel(logging.DEBUG)
            info_handler.setFormatter(formatter)
            
            error_path = f"{str(base_file_name)}_error.log"
            error_handler = logging.FileHandler(error_path)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str, exc_info : bool = True) -> None:
        self.logger.error(message, exc_info)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)


def get_base_transformations(img_size):
    """Standard preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])