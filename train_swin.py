import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil

from package.config import (
    DATA_DIR, DEVICE, EPOCHS, NFOLDS, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
    TEST_SPLIT, OPTIMIZE_METRIC, MODEL_NAME, IMG_SIZE,
    TEMP_WEIGHTS_PATH, CLASS_NAMES_PATH,WEIGHTS_PATH
)
from package.utils import FullDataset, Logger, get_base_transformations
from package.model import ProgressiveClassifier


def train(clf_type : str):
    logger = Logger("swin_train", file_name="swin_train")
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Classifier : {clf_type}")
    logger.info(f"Device     : {DEVICE}")
    logger.info(f"Epochs : {EPOCHS}  |  Folds : {NFOLDS}  |  BS : {BATCH_SIZE}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    transform = get_base_transformations(IMG_SIZE)

    logger.info(f"Loading dataset : {DATA_DIR}  (img_size={IMG_SIZE})")
    full_dataset = FullDataset(DATA_DIR, transform)

    targets     = np.array(full_dataset.targets)
    class_names = full_dataset.classes

    logger.info(f"Classes : {class_names}")
    logger.info(f"Samples : {len(targets)}")

    CLASS_NAMES_PATH.write_text("\n".join(class_names))
    logger.info(f"Class names saved -> {CLASS_NAMES_PATH}")

    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_names) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    logger.info("Class weights:")
    for name, w in zip(class_names, class_weights):
        logger.info(f"  {name}: {w:.4f}")

    # ── Train / test split ───────────────────────────────────────────────────
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=TEST_SPLIT,
        stratify=targets,
        random_state=42,
    )
    logger.info(f"Split : {len(train_val_idx)} train/val  |  {len(test_idx)} test")

    skf        = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    tv_targets = targets[train_val_idx]

    fold_results    = []
    best_fold_val   = 0.0

    # ── K-Fold ───────────────────────────────────────────────────────────────
    for fold, (rel_tr, rel_val) in enumerate(skf.split(train_val_idx, tv_targets)):
        tr_idx  = train_val_idx[rel_tr]
        val_idx = train_val_idx[rel_val]

        def _loader(indices, shuffle):
            return DataLoader(
                Subset(full_dataset, indices),
                batch_size=BATCH_SIZE, shuffle=shuffle,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
            )

        train_loader = _loader(tr_idx,  True)
        val_loader   = _loader(val_idx, False)

        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{NFOLDS}")
        logger.info(f"{'='*70}")

        clf = ProgressiveClassifier(class_weights_tensor=class_weights_tensor,)

        clf.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            use_sam=False,
        )

        fold_results.append({
            "fold":           fold + 1,
            f"val_{OPTIMIZE_METRIC}": clf.best_metric_value,
            "val_acc":        clf.best_acc,
            "val_recall":     clf.best_recall,
            "val_f1":         clf.best_f1,
        })

        if clf.best_metric_value > best_fold_val:
            best_fold_val = clf.best_metric_value
            shutil.copy(str(TEMP_WEIGHTS_PATH), str(WEIGHTS_PATH))  # promote to global best
            logger.info(f"  * New global best ({best_fold_val:.4f}) -- global checkpoint updated")
        
        # Cleanup temp checkpoint
        TEMP_WEIGHTS_PATH.unlink(missing_ok=True)


        del clf, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(fold_results)
    logger.info("\nK-Fold Summary:\n" + df.to_string(index=False))
    col = f"val_{OPTIMIZE_METRIC}"
    logger.info(f"Mean {OPTIMIZE_METRIC}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\nFinal held-out test evaluation...")

    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )

    eval_clf = ProgressiveClassifier(class_weights_tensor=class_weights_tensor,)
    checkpoint = WEIGHTS_PATH
    eval_clf.load(str(checkpoint))
    logger.info(f"Loaded checkpoint : {checkpoint}")

    metrics = eval_clf.evaluate(test_loader, class_names)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"\nWeights saved -> {checkpoint}")
    logger.info("Done.")


if __name__ == "__main__":
    train()