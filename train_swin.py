import os
import argparse
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path

from package.config import (
    DATA_DIR, DEVICE, EPOCHS, NFOLDS, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, LR,
    TEST_SPLIT, OPTIMIZE_METRIC, MODEL_NAME, IMG_SIZE,
    CLASS_NAMES_PATH, WEIGHTS_PATH, TEMP_WEIGHTS_PATH,
    USE_MLFLOW, MLFLOW_TRACKING_URI, SAVE_DIR
)
from package.utils import FullDataset, Logger, get_base_transformations
from package.model import ProgressiveClassifier

CLASSIFIER_TYPE = "progressive"


if "SM_CHANNEL_TRAIN" in os.environ:
    SM_MODEL_DIR   = os.environ.get("SM_MODEL_DIR")
    SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN")
    import package.config as cfg
    
    cfg.DATA_DIR           = SM_CHANNEL_TRAIN
    cfg.SAVE_DIR           = Path(SM_MODEL_DIR)
    cfg.OUTPUT_DIR         = Path(SM_MODEL_DIR)
    cfg.WEIGHTS_PATH       = Path(SM_MODEL_DIR) / "swin_model_weights.pth"
    cfg.CLASS_NAMES_PATH   = Path(SM_MODEL_DIR) / "swin_class_names.txt"
    cfg.CLASS_WEIGHTS_PATH = Path(SM_MODEL_DIR) / "class_weights.npy"
    cfg.TEMP_WEIGHTS_PATH  = Path(SM_MODEL_DIR) / "temp_model_weights.pth"
    cfg.LOG_DIR            = Path(SM_MODEL_DIR) / "logs"
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    DATA_DIR           = cfg.DATA_DIR
    SAVE_DIR           = cfg.SAVE_DIR
    OUTPUT_DIR         = cfg.OUTPUT_DIR
    WEIGHTS_PATH       = cfg.WEIGHTS_PATH
    CLASS_NAMES_PATH   = cfg.CLASS_NAMES_PATH
    CLASS_WEIGHTS_PATH = cfg.CLASS_WEIGHTS_PATH
    TEMP_WEIGHTS_PATH  = cfg.TEMP_WEIGHTS_PATH
    LOG_DIR            = cfg.LOG_DIR

    print(f"[SageMaker] DATA_DIR  = {DATA_DIR}")
    print(f"[SageMaker] SAVE_DIR  = {SAVE_DIR}")
if "OVERRIDE_EPOCHS" in os.environ:
    cfg.EPOCHS = int(os.environ["OVERRIDE_EPOCHS"])
    print(f"[SageMaker] EPOCHS  = {cfg.EPOCHS}")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Path to dataset root (overrides config.DATA_DIR)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config.EPOCHS)")
    parser.add_argument("--folds",  type=int, default=None,
                        help="Number of CV folds (overrides config.NFOLDS)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Number of epochs for Early Stopping (overrides config.PATIENCE)")
    return parser.parse_args()


def configure(
    data_dir:     str  = None,
    epochs:       int  = None,
    nfolds:       int  = None,
    patience:     int  = None,
):
    import package.config as cfg
    global DATA_DIR, EPOCHS, NFOLDS

    if data_dir is not None:
        cfg.DATA_DIR = data_dir
        DATA_DIR = data_dir

    if epochs is not None:
        cfg.EPOCHS = epochs
        EPOCHS = epochs

    if nfolds is not None:
        cfg.NFOLDS = nfolds
        NFOLDS = nfolds

    if patience is not None:
        cfg.PATIENCE = patience


def _build_loader(full_dataset, indices, shuffle):
    return DataLoader(
        Subset(full_dataset, indices),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )


def train():
    args = _parse_args()

    configure(
        data_dir=args.data_dir,
        epochs=args.epochs,
        nfolds=args.folds,
        patience=args.patience,
    )

    logger = Logger("swin_train", file_name="swin_train")
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Classifier : {CLASSIFIER_TYPE}")
    logger.info(f"Device     : {DEVICE}")
    logger.info(f"Epochs     : {EPOCHS}  |  Folds : {NFOLDS}  |  BS : {BATCH_SIZE}")

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
    for item in enumerate(skf.split(train_val_idx, tv_targets)) :
        fold, (rel_tr, rel_val) = item

        logger.info(f"\n{'=' * 70}")
        logger.info(f"FOLD {fold + 1}/{NFOLDS}")
        logger.info(f"{'=' * 70}")

        tr_idx  = train_val_idx[rel_tr]
        val_idx = train_val_idx[rel_val]
        train_loader = _build_loader(full_dataset, tr_idx,  shuffle=True)
        val_loader   = _build_loader(full_dataset, val_idx, shuffle=False)

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

    test_loader = _build_loader(full_dataset, test_idx, shuffle=False)

    eval_clf = ProgressiveClassifier(class_weights_tensor=class_weights_tensor,)
    checkpoint = WEIGHTS_PATH if WEIGHTS_PATH.exists() else TEMP_WEIGHTS_PATH
    eval_clf.load(str(checkpoint))
    logger.info(f"Loaded checkpoint : {checkpoint}")

    metrics = eval_clf.evaluate(test_loader, class_names)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")

    # MLflow log if enabled
    if USE_MLFLOW :
        try:
            from package.mlflow_tracker import log_cross_validation_result

            experiment_name = f"{MODEL_NAME}_classifier={CLASSIFIER_TYPE}_metric={OPTIMIZE_METRIC}_ADNI_cross_validation"
            log_cross_validation_result(
                model_name=MODEL_NAME,
                classifier_type=CLASSIFIER_TYPE,
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
                experiment_name=experiment_name,
            )
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    logger.info(f"\nWeights saved -> {checkpoint}")
    logger.info("Done.")


if __name__ == "__main__":
    train()