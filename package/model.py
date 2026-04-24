import torch
import torch.optim as optim
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm

from package.loss import FocalLoss
from package.optimizer import SAM
from package.layer_groups import get_swin_groups
from package.config import (
    MODEL_NAME, NUM_CLASSES, DEVICE, PATIENCE,
    OPTIMIZE_METRIC, EPOCHS, LR, MIN_DELTA,
    TEMP_WEIGHTS_PATH, CLASS_WEIGHTS_PATH,
)


class ProgressiveClassifier :
    def __init__(self, class_weights_tensor : torch.FloatTensor = None):
        self.class_weights_tensor = class_weights_tensor
        np.save(str(CLASS_WEIGHTS_PATH), class_weights_tensor.cpu().numpy())
        
        # Model (built by subclass)
        self.model = None
        
        # Training state
        self.best_metric_value = 0.0
        self.best_recall = 0.0
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.history = []
        
        # Build model
        self.build_model()
        
        if self.model is None:
            raise ValueError("build_model() must set self.model")
        
        self.model = self.model.to(DEVICE)
    
    def build_model(self):
        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=NUM_CLASSES
        )
        
        # Get layer groups for discriminative LRs
        self.layer_groups = get_swin_groups(self.model)
        
    def forward(self, images):
        return self.model(images)
    
    def get_predictions(self, outputs) -> torch.Tensor:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Bias logits by adding log(weights) to improve recall for high-weight classes
        # This works because Softmax(x + log(w)) = w * exp(x) / sum(w * exp(x))
        if self.class_weights_tensor.device != outputs.device:
            self.class_weights_tensor = self.class_weights_tensor.to(outputs.device)
        
        # Use log-weights to bias logits (safe with small epsilon)
        log_weights = torch.log(self.class_weights_tensor + 1e-10)
        outputs = outputs + log_weights.view(1, -1)
            
        # return torch.argmax(outputs, dim=1)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds
    
    def compute_loss(self, outputs, labels):
        if not hasattr(self, 'focal_loss'):
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, weights=self.class_weights_tensor).to(DEVICE)
        return self.focal_loss(outputs, labels)
    
    def _get_metric_value(self, labels: List, preds: List, metric: str) -> float:
        if metric == 'recall':
            return recall_score(labels, preds, average='macro', zero_division=0)
        elif metric == 'accuracy':
            return accuracy_score(labels, preds) * 100
        elif metric == 'f1':
            return f1_score(labels, preds, average='macro', zero_division=0)
        elif metric == 'precision':
            return precision_score(labels, preds, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _get_discriminative_params(self, base_lr):
        lr_multipliers = [1/100, 1/10, 1/3, 1.0, 10.0]
        
        param_groups = []
        for params, mult in zip(self.layer_groups, lr_multipliers):
            if params:
                param_groups.append({
                    'params': params,
                    'lr': base_lr * mult
                })
        
        return param_groups
    
    def fit(self, train_loader, val_loader, use_sam = True):
        print(f"\n{'='*80}")
        print(f"PROGRESSIVE FINE-TUNING: {MODEL_NAME}")
        print(f"Optimizing for: {OPTIMIZE_METRIC.upper()}")
        print(f"{'='*80}\n")
        remaining_epochs = EPOCHS
        
        # Phase 1: Classifier only (5 epochs)
        print("="*80)
        print("PHASE 1: Training Classifier Only (Backbone Frozen)")
        print("="*80)
        
        self._train_phase(
            phase=1,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=min(5, remaining_epochs),
            lr=LR * 10,  # Higher LR for random classifier
            freeze_mode='classifier_only',
            use_sam=False,
            patience=5,
        )
        
        # Phase 2: Top 50% layers (10 epochs)
        remaining_epochs = max(0, remaining_epochs - 5)
        if remaining_epochs > 0:
            print("\n" + "="*80)
            print("PHASE 2: Fine-tuning Top Layers (Bottom 50% Frozen)")
            print("="*80)
            
            self._train_phase(
                phase=2,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=min(10, remaining_epochs),
                lr=LR,
                freeze_mode='top_50',
                use_sam=False,
                patience=10,
            )
        
        # Phase 3: All layers with discriminative LRs (remaining epochs)
        remaining_epochs = max(0, remaining_epochs - 10)
        if remaining_epochs > 0:
            print("\n" + "="*80)
            print("PHASE 3: Discriminative Fine-Tuning (All Layers)")
            print("="*80)
            
            self._train_phase(
                phase=3,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=remaining_epochs,
                lr=LR,
                freeze_mode='all_discriminative',
                use_sam=use_sam,  # SAM only in phase 3
                patience=PATIENCE,
            )
        
        print(f"\n{'='*80}")
        print("PROGRESSIVE FINE-TUNING COMPLETE")
        print(f"Final Best {OPTIMIZE_METRIC.capitalize()}: {self.best_metric_value:.4f} ★")
        print(f"Final Best Recall: {self.best_recall:.4f}")
        print(f"Final Best Accuracy: {self.best_acc:.2f}%")
        print(f"{'='*80}\n")
        
        return self.history
    
    def _train_phase(self, phase, train_loader, val_loader, epochs, lr, freeze_mode, use_sam, patience):
        
        # Freeze/unfreeze according to mode
        if freeze_mode == 'classifier_only':
            # Freeze all except classifier
            for param in self.model.parameters():
                param.requires_grad = False
            if self.layer_groups[-1]:
                for param in self.layer_groups[-1]:
                    param.requires_grad = True
                    
        elif freeze_mode == 'top_50':
            # Unfreeze top 50%
            for param in self.model.parameters() :
                param.requires_grad = False
            top_groups = self.layer_groups[2:]  # Groups 2, 3, 4
            for group in top_groups:
                for param in group:
                    param.requires_grad = True
                
        elif freeze_mode == 'all_discriminative':
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Create optimizer
        if freeze_mode == 'all_discriminative':
            # Discriminative LRs
            param_groups = self._get_discriminative_params(lr)
            print(f"Discriminative LR groups:")
            for i, group in enumerate(param_groups):
                print(f"  Group {i}: {len(list(group['params']))} params, LR={group['lr']:.2e}")
        else:
            # Single LR
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if use_sam:
            optimizer = SAM(param_groups, optim.AdamW, lr=lr, weight_decay=0.01, rho=0.05)
        else:
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer.base_optimizer if use_sam else optimizer,
            T_0 = (epochs // 7) + 1,
            T_mult = 2,
            eta_min = 1e-7
        )
        
        # Scaler
        scaler = torch.amp.GradScaler(enabled=(DEVICE == 'cuda' and not use_sam))
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_recall = self.train_epoch(
                train_loader, optimizer, scaler if not use_sam else None
            )
            
            # Validate
            val_loss, val_acc, val_recall, val_prec, val_f1, primary_value, per_class_recall = self.validate_epoch(val_loader)

            # Record history
            self.history.append({
                'phase'                  : phase,
                'epoch'                  : epoch + 1,
                'train_loss'             : train_loss,
                'train_acc'              : train_acc,
                'train_recall'           : train_recall,
                'val_loss'               : val_loss,
                'val_acc'                : val_acc,
                'val_recall'             : val_recall,
                'val_precision'          : val_prec,
                'val_f1'                 : val_f1,
                'val_per_class_recall'   : per_class_recall,
                f'val_{OPTIMIZE_METRIC}' : primary_value
            })
            
            # Check improvement
            improved = False
            if primary_value > self.best_metric_value + MIN_DELTA:
                self.best_metric_value = primary_value
                improved = True
            if val_recall > self.best_recall:
                self.best_recall = val_recall
            if val_acc > self.best_acc:
                self.best_acc = val_acc
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
            
            # Print
            if improved:
                print(f"  [Epoch {epoch+1}/{epochs}] {OPTIMIZE_METRIC}: {primary_value:.4f} ★, "
                      f"Acc: {val_acc:.2f}%, Recall: {val_recall:.4f}")
                self.best_epoch = epoch + 1
                self.save(str(TEMP_WEIGHTS_PATH))
                patience_counter = 0
            else:
                print(f"  [Epoch {epoch+1}/{epochs}] {OPTIMIZE_METRIC}: {primary_value:.4f}, "
                      f"Acc: {val_acc:.2f}%")
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break
            
        print(f"Phase {phase} Complete - Best {OPTIMIZE_METRIC}: {self.best_metric_value:.4f}")
    
    def train_epoch(self, train_loader, optimizer, scaler=None):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        # Check if SAM optimizer
        is_sam = isinstance(optimizer, SAM)
        use_amp = not is_sam and DEVICE == 'cuda'
        
        print("Training")
        for images, labels in train_loader :
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if is_sam:
                # SAM: Two-step (no AMP)
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard (with AMP if available)
                if use_amp and scaler:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.forward(images)
                        loss = self.compute_loss(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.forward(images)
                    loss = self.compute_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Metrics
            running_loss += loss.detach().item() * images.size(0)
            
            with torch.no_grad():
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, recall
    
    def validate_epoch(self, val_loader, primary_metric : str = "recall"):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.inference_mode():
            print("Validating")
            for images, labels in val_loader :
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate all metrics
        avg_loss         = running_loss / len(val_loader.dataset)
        acc              = accuracy_score(all_labels, all_preds) * 100
        recall           = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision        = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1               = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        primary_value    = self._get_metric_value(all_labels, all_preds, primary_metric)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        return avg_loss, acc, recall, precision, f1, primary_value, per_class_recall
    
    def evaluate(self, test_loader, class_names: Optional[List[str]] = None):
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.inference_mode():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.forward(images)
                probs = torch.softmax(outputs, dim=1)
                preds = self.get_predictions(outputs)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate all metrics
        acc       = accuracy_score(all_labels, all_preds) * 100
        recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Per-class metrics
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        cm               = confusion_matrix(all_labels, all_preds)
        report           = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        return {
            'accuracy'         : acc,
            'recall'           : recall,
            'precision'        : precision,
            'f1'               : f1,
            'per_class_recall' : per_class_recall,
            'confusion_matrix' : cm,
            'report'           : report,
            'labels'           : all_labels,
            'preds'            : all_preds,
            'probs'            : all_probs
        }
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))