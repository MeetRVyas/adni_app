"""
Training script for swin_base_patch4_window7_224.ms_in22k_ft_in1k
using ProgressiveClassifier from the existing module.

Respects config.USE_BDA_PIPELINE:
  True  → data loaded via DiseaseRegistry / WebDataset shards
  False → data loaded via legacy FullDataset / ImageFolder

Usage:
    python train_swin.py
    python train_swin.py --no_bda          # force legacy mode
    python train_swin.py --disease adni    # explicit disease tag

Outputs:
    saved_models/swin_progressive_best.pth   -- model weights
    saved_models/swin_class_names.txt        -- class names (one per line)
"""

import argparse
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold


from package.config import (
    DATA_DIR, DEVICE, EPOCHS, NFOLDS, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, LR,
    TEST_SPLIT, OPTIMIZE_METRIC, MODEL_NAME, IMG_SIZE,
    ROOT, CLASS_NAMES_PATH, WEIGHTS_PATH, TEMP_WEIGHTS_PATH,
    USE_BDA_PIPELINE, DISEASE_ID, REGISTRY_DIR, SHARDS_DIR,
    DATASET_LAYOUT, DATASET_CSV, AUTO_RUN_PIPELINE,
    USE_MLFLOW, MLFLOW_TRACKING_URI,
)
from package.utils import FullDataset, Logger, get_base_transformations, ensure_pipeline_ready
from package.model import ProgressiveClassifier
from data_pipeline.loaders.webdataset_loader import get_dataloader

CLASSIFIER_TYPE = "progressive"

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", type=str, default="adni",
                   help="Disease tag (overrides config.DISEASE_ID)")
    parser.add_argument("--no_bda", action="store_true",
                   help="Force legacy FullDataset mode")
    
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Path to dataset root (overrides config.DATA_DIR)")

    # --- Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config.EPOCHS)")
    parser.add_argument("--folds",  type=int, default=None,
                        help="Number of CV folds (overrides config.NFOLDS)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Number of epochs for Early Stopping (overrides config.PATIENCE)")
    return parser.parse_args()


def _build_bda_loaders(disease_id, img_size, fold_idx):
    """Return (train_loader, val_loader) from BDA pipeline for one fold."""

    train_loader = get_dataloader(
        disease=disease_id, fold=fold_idx, split="train",
        img_size=img_size, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
        shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )
    val_loader = get_dataloader(
        disease=disease_id, fold=fold_idx, split="val",
        img_size=img_size, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
        shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )
    return train_loader, val_loader


def _build_legacy_loader(full_dataset, indices, shuffle):
    return DataLoader(
        Subset(full_dataset, indices),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )

def configure(
    data_dir:     str  = None,
    epochs:       int  = None,
    nfolds:       int  = None,
    patience:     int  = None,
):
    """
    Override mutable config values at runtime.
    Only non-None arguments are applied.
    """
    import package.config as cfg

    if data_dir is not None:
        cfg.DATA_DIR = data_dir

    if epochs is not None:
        cfg.EPOCHS = epochs

    if nfolds is not None:
        cfg.NFOLDS = nfolds

    if patience is not None:
        cfg.PATIENCE = patience

def train():
    args = _parse_args()

    configure(
        data_dir=args.data_dir,
        epochs=args.epochs,
        nfolds=args.folds,
        patience=args.patience,
    )

    disease_id  = args.disease or DISEASE_ID
    use_bda     = USE_BDA_PIPELINE and not args.no_bda

    logger = Logger("swin_train", file_name="swin_train")
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Classifier : {CLASSIFIER_TYPE}")
    logger.info(f"Device     : {DEVICE}")
    logger.info(f"Epochs     : {EPOCHS}  |  Folds : {NFOLDS}  |  BS : {BATCH_SIZE}")
    logger.info(f"BDA mode   : {use_bda}  |  Disease : {disease_id}")

    # ── Resolve classes and class weights ─────────────────────────────────────
    if use_bda:
        pipeline_ready = ensure_pipeline_ready(
            disease_id=disease_id,
            data_dir=DATA_DIR,
            layout=DATASET_LAYOUT,
            registry_dir=REGISTRY_DIR,
            shards_dir=SHARDS_DIR,
            nfolds=NFOLDS,
            csv_path=DATASET_CSV,
            auto_run=AUTO_RUN_PIPELINE,
        )
        if not pipeline_ready:
            logger.warning("BDA registry unavailable — falling back to legacy mode.")
            use_bda = False

    if use_bda:
        from data_pipeline.registry.disease_registry import DiseaseRegistry
        reg = DiseaseRegistry(REGISTRY_DIR, use_dask=False)
        class_names = reg.class_names(disease_id)
        class_weights_np = reg.class_weights(disease_id, split="train")
        class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)
    else:
        transform = get_base_transformations(IMG_SIZE)
        full_dataset = FullDataset(DATA_DIR, transform)
        targets = np.array(full_dataset.targets)
        class_names = full_dataset.classes
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        class_weights = total_samples / (len(class_names) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

        train_val_idx, test_idx = train_test_split(
            np.arange(len(targets)),
            test_size=TEST_SPLIT, stratify=targets, random_state=42,
        )
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        tv_targets = targets[train_val_idx]

    logger.info(f"Classes ({len(class_names)}): {class_names}")
    CLASS_NAMES_PATH.write_text("\n".join(class_names))
    logger.info(f"Class names saved -> {CLASS_NAMES_PATH}")

    fold_results  = []
    best_fold_val = 0.0

    # ── K-Fold ────────────────────────────────────────────────────────────────
    if use_bda:
        fold_iter = range(NFOLDS)
    else:
        fold_iter = enumerate(skf.split(train_val_idx, tv_targets))

    for item in fold_iter:
        if use_bda:
            fold = item
        else:
            fold, (rel_tr, rel_val) = item

        logger.info(f"\n{'=' * 70}")
        logger.info(f"FOLD {fold + 1}/{NFOLDS}")
        logger.info(f"{'=' * 70}")

        if use_bda:
            train_loader, val_loader = _build_bda_loaders(disease_id, IMG_SIZE, fold)
        else:
            tr_idx  = train_val_idx[rel_tr]
            val_idx = train_val_idx[rel_val]
            train_loader = _build_legacy_loader(full_dataset, tr_idx,  shuffle=True)
            val_loader   = _build_legacy_loader(full_dataset, val_idx, shuffle=False)

        clf = ProgressiveClassifier(class_weights_tensor=class_weights_tensor,)

        clf.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            use_sam=False,
        )

        fold_results.append({
            "fold":                       fold + 1,
            f"val_{OPTIMIZE_METRIC}":     clf.best_metric_value,
            "val_acc":                    clf.best_acc,
            "val_recall":                 clf.best_recall,
            "val_f1":                     clf.best_f1,
        })

        if clf.best_metric_value > best_fold_val:
            best_fold_val = clf.best_metric_value
            clf.save(str(WEIGHTS_PATH))
            logger.info(f"  * New best fold ({best_fold_val:.4f}) — fold checkpoint updated")

        del clf, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(fold_results)
    logger.info("\nK-Fold Summary:\n" + df.to_string(index=False))
    col = f"val_{OPTIMIZE_METRIC}"
    logger.info(f"Mean {OPTIMIZE_METRIC}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\nFinal held-out test evaluation...")

    if use_bda:
        from data_pipeline.loaders.webdataset_loader import get_dataloader
        test_loader = get_dataloader(
            disease=disease_id, fold=-1, split="test",
            img_size=IMG_SIZE, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, registry_dir=REGISTRY_DIR,
            shards_dir=SHARDS_DIR, pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        )
    else:
        test_loader = _build_legacy_loader(full_dataset, test_idx, shuffle=False)

    eval_clf = ProgressiveClassifier(class_weights_tensor=class_weights_tensor,)
    checkpoint = WEIGHTS_PATH if WEIGHTS_PATH.exists() else TEMP_WEIGHTS_PATH
    eval_clf.load(str(checkpoint))
    logger.info(f"Loaded checkpoint : {checkpoint}")

    experiment_name = f"{MODEL_NAME}_classifier={CLASSIFIER_TYPE}_metric={OPTIMIZE_METRIC}"
    if use_bda:
        experiment_name += f"_disease={disease_id}"

    metrics = eval_clf.evaluate(test_loader, class_names)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")

    # MLflow log if enabled
    if USE_MLFLOW and use_bda:
        try:
            from data_pipeline.tracking.mlflow_tracker import log_cross_validation_result
            log_cross_validation_result(
                model_name=MODEL_NAME,
                classifier_type=CLASSIFIER_TYPE,
                disease=disease_id,
                fold_metrics=fold_results,
                final_metrics={
                    'test_accuracy':  metrics['accuracy'],
                    'test_recall':    metrics['recall'],
                    'test_precision': metrics['precision'],
                    'test_f1':        metrics['f1'],
                },
                config={'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR},
                model_path=str(checkpoint),
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment_name=f"{disease_id}_cross_validation",
            )
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.info(f"\nWeights saved -> {checkpoint}")
    logger.info("Done.")


if __name__ == "__main__":
    train()