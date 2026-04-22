"""
huggingface_adapter.py
======================
Streams MedMNIST (and other HuggingFace datasets) to local PNG files
without downloading the entire dataset first.

Key datasets supported
----------------------
  albertvillanova/medmnist           — MedMNIST v2 (canonical HF upload)
    Subsets: pathmnist, dermamnist, octmnist, pneumoniamnist,
             retinamnist, breastmnist, bloodmnist, tissuemnist,
             organamnist, organcmnist, organsmnist

Usage
-----
    from data_pipeline.ingestion.huggingface_adapter import HuggingFaceAdapter

    adapter = HuggingFaceAdapter(
        hf_dataset="albertvillanova/medmnist",
        subset="retinamnist",
        output_dir="staging/retinamnist_pngs",
        target_size=224,
    )
    adapter.fetch()
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


class HuggingFaceAdapter:
    """
    Streams a HuggingFace dataset and saves each sample as a PNG.

    Parameters
    ----------
    hf_dataset : str
        HuggingFace dataset identifier, e.g. 'albertvillanova/medmnist'.
    subset : str | None
        Dataset configuration / subset name, e.g. 'retinamnist'.
    output_dir : str
        Where to save the PNGs.
    target_size : int
        Side length for bicubic upscaling (0 = no resize).
    image_col : str
        Column name in the HF dataset that holds the image.
    label_col : str
        Column name for the label.
    splits : list[str]
        Which splits to download.
    """

    CANONICAL_MEDMNIST_REPO = "albertvillanova/medmnist"

    def __init__(
        self,
        hf_dataset: str = "albertvillanova/medmnist",
        subset: Optional[str] = "retinamnist",
        output_dir: str = "staging/medmnist",
        target_size: int = 224,
        image_col: str = "image",
        label_col: str = "label",
        splits: Optional[list[str]] = None,
    ):
        self.hf_dataset  = hf_dataset
        self.subset      = subset
        self.output_dir  = Path(output_dir)
        self.target_size = target_size
        self.image_col   = image_col
        self.label_col   = label_col
        self.splits      = splits or ["train", "validation", "test"]

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self) -> Path:
        """Stream dataset and save all images as PNGs. Returns output_dir."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required.\n"
                "Install: pip install datasets"
            )

        marker = self.output_dir / ".hf_download_complete"
        if marker.exists():
            print(f"[HuggingFaceAdapter] Already fetched → {self.output_dir}")
            return self.output_dir

        print(f"[HuggingFaceAdapter] Loading {self.hf_dataset}/{self.subset} …")
        ds_kwargs = {}
        if self.subset:
            ds_kwargs["name"] = self.subset

        dataset = load_dataset(self.hf_dataset, **ds_kwargs, trust_remote_code=True)

        total_saved = 0
        for split in self.splits:
            hf_split = "validation" if split == "val" else split
            if hf_split not in dataset:
                print(f"  [WARN] Split '{hf_split}' not in dataset, skipping")
                continue

            split_data   = dataset[hf_split]
            save_split   = "val" if hf_split == "validation" else hf_split
            split_saved  = self._save_split(split_data, save_split)
            total_saved += split_saved
            print(f"  [{split}] {split_saved} images saved")

        marker.touch()
        print(f"[HuggingFaceAdapter] Done. {total_saved} total images → {self.output_dir}")
        return self.output_dir

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_split(self, split_data, split_name: str) -> int:
        saved = 0
        for idx, sample in enumerate(split_data):
            img  = sample[self.image_col]
            lbl  = sample[self.label_col]

            # HF image may be PIL.Image or raw bytes or numpy
            pil_img = self._to_pil(img)

            # Upscale if needed
            if self.target_size > 0:
                target = (self.target_size, self.target_size)
                if pil_img.size != target:
                    pil_img = pil_img.resize(target, Image.BICUBIC)

            # Ensure RGB
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Save
            label_str = str(int(lbl) if not isinstance(lbl, list) else lbl[0])
            out_dir   = self.output_dir / split_name / label_str
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path  = out_dir / f"img_{idx:08d}.png"
            pil_img.save(str(out_path), format="PNG")
            saved += 1

        return saved

    @staticmethod
    def _to_pil(img) -> Image.Image:
        """Convert various image representations to PIL.Image."""
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                lo, hi = img.min(), img.max()
                if hi > lo:
                    img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            return Image.fromarray(img)
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img))
        raise TypeError(f"Cannot convert type {type(img)} to PIL.Image")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: download all MedMNIST subsets
# ─────────────────────────────────────────────────────────────────────────────

MEDMNIST_SUBSETS = [
    "pathmnist", "dermamnist", "octmnist", "pneumoniamnist",
    "retinamnist", "breastmnist", "bloodmnist", "tissuemnist",
    "organamnist", "organcmnist", "organsmnist",
]


def download_all_medmnist(
    output_root: str = "staging/medmnist",
    target_size: int = 224,
    subsets: Optional[list[str]] = None,
):
    """
    Download all (or selected) MedMNIST subsets from HuggingFace.

    Parameters
    ----------
    output_root : str
        Parent directory; each subset gets its own subdirectory.
    target_size : int
        Image resize target.
    subsets : list | None
        If None, download all subsets.
    """
    chosen = subsets or MEDMNIST_SUBSETS
    for subset in chosen:
        adapter = HuggingFaceAdapter(
            hf_dataset=HuggingFaceAdapter.CANONICAL_MEDMNIST_REPO,
            subset=subset,
            output_dir=os.path.join(output_root, subset),
            target_size=target_size,
        )
        try:
            adapter.fetch()
        except Exception as e:
            print(f"[download_all_medmnist] Failed for {subset}: {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="albertvillanova/medmnist")
    p.add_argument("--subset",      default="retinamnist")
    p.add_argument("--output_dir",  default="staging/medmnist/retinamnist")
    p.add_argument("--target_size", type=int, default=224)
    args = p.parse_args()

    adapter = HuggingFaceAdapter(
        hf_dataset=args.dataset,
        subset=args.subset,
        output_dir=args.output_dir,
        target_size=args.target_size,
    )
    adapter.fetch()
