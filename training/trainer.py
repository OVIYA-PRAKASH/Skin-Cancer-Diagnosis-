"""
Phase 10 & 11: Training and Validation Pipeline.
Implements the MultiModalTrainer engine that orchestrates training epochs with Mixed Precision (AMP),
class-weighted CrossEntropy loss, AdamW optimization, CosineAnnealingLR scheduling,
gradient clipping, and validation checkpointing.
"""

import os
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# We import metrics calculation from sklearn for validation tracking
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger("SkinCancerAI.Trainer")


class MultiModalTrainer:
    """
    Engine that orchestrates multi-modal training and validation loops.
    Handles device mapping, mixed-precision training, weight decay, and saving checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor],
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 20,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        max_grad_norm: float = 1.0
    ):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The HybridMultiModalClassifier model instance.
            train_loader (DataLoader): PyTorch DataLoader yielding training batches.
            val_loader (DataLoader): PyTorch DataLoader yielding validation batches.
            class_weights (Optional[torch.Tensor]): 7-element tensor containing class weights.
            device (torch.device): target device (cuda or cpu).
            learning_rate (float): Initial optimizer learning rate.
            weight_decay (float): Decoupled weight decay multiplier for AdamW.
            epochs (int): Number of epochs to train.
            checkpoint_dir (str): Folder where model checkpoints will be stored.
            use_amp (bool): If True, uses Automatic Mixed Precision (float16).
            max_grad_norm (float): Gradient clipping threshold.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 1. Setup Loss function: use class-weighted Cross Entropy to solve class imbalance
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

        # 2. Setup Optimizer: AdamW handles Vision Transformer weight decay better than standard Adam
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 3. Setup Scheduler: Cosine Annealing decays the learning rate smoothly to a minimum
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        # 4. Setup AMP GradScaler (Only active on CUDA GPU, helps prevent underflow in float16)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.device.type == "cuda")

        # Track learning history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }
        
        self.best_f1 = 0.0
        logger.info(
            f"Trainer initialized [AMP: {self.use_amp} | Weight Decay: {weight_decay} | Epochs: {self.epochs}]"
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Runs a single training epoch over all dataloader mini-batches.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, metadata, labels) in enumerate(self.train_loader):
            # Send batch data to the target device (GPU or CPU)
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass under Automatic Mixed Precision autocast context
            with torch.amp.autocast('cuda', enabled=self.use_amp and self.device.type == "cuda"):
                # Forward pass
                logits = self.model(images, metadata)
                loss = self.criterion(logits, labels)

            # Backward pass and optimizer step using GradScaler
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                # Unscale gradients before clipping to ensure correct thresholds
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping to prevent gradient explosion
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] | Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                    f"Loss: {loss.item():.4f} | Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> Tuple[float, float, float]:
        """
        Evaluates the model on the validation dataset.

        Returns:
            Tuple[float, float, float]: (average_validation_loss, accuracy, macro_f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, metadata, labels in self.val_loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp and self.device.type == "cuda"):
                    logits = self.model(images, metadata)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        # Macro F1 is standard for publication to account for rare class classification accuracy
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return avg_loss, accuracy, macro_f1

    def fit(self) -> Dict[str, List[float]]:
        """
        Runs the full training process for the specified number of epochs,
        applies learning rate decays, and saves best/last model checkpoints.

        Returns:
            Dict[str, List[float]]: History of training/validation losses and metrics.
        """
        logger.info(f"Starting model training for {self.epochs} epochs...")
        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # 1. Run training epoch
            train_loss = self.train_epoch(epoch)
            
            # 2. Run validation epoch
            val_loss, val_acc, val_f1 = self.validate()

            # 3. Step scheduler to decay learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Save stats to history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            epoch_elapsed = time.time() - epoch_start
            logger.info(
                f"[Epoch {epoch} Complete] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f} | LR: {current_lr:.6f} | Time: {epoch_elapsed:.1f}s"
            )

            # 4. Checkpoint saving: Save best model if validation Macro-F1 improves
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "best_f1": self.best_f1
            }

            # Save the latest model checkpoint (allows resuming training)
            last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
            torch.save(checkpoint_state, last_path)

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                checkpoint_state["best_f1"] = self.best_f1
                torch.save(checkpoint_state, best_path)
                logger.info(f"[OK] New Best Model Saved! Validation Macro-F1: {val_f1:.4f}")

        total_elapsed = time.time() - total_start
        logger.info(
            f"Training session complete! Total Time: {total_elapsed/60:.2f} mins. "
            f"Best Validation Macro-F1 reached: {self.best_f1:.4f}"
        )
        return self.history
