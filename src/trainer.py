import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from timm.data import Mixup

class Trainer:
    def __init__(
        self, 
        args,
        model: nn.Module,
        model_name: str,
        device: torch.device,  
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_fn: nn.Module,
        result_path: str, 
        mixup_args: dict
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.result_path = args.model_dir
        self.epochs = args.epochs
        self.best_models = []
        self.lowest_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.model_name = args.model_name

        self.mixup_fn = Mixup(**mixup_args) if mixup_args else None # Mixup & CutMix 적용

    def save_model(self, epoch: int, loss: float) -> None:
        os.makedirs(self.result_path, exist_ok=True) # 모델 저장경로 설정
        
        # 현재 모델 저장
        current_model_path = os.path.join(self.result_path, f'{self.model_name}_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)
        
        # 최상위 3개 모델 저장
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1) 
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 loss 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, f'{self.model_name}.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch+1}epoch result: Loss = {loss:.4f}")

    # 한 에폭 동안 학습을 수행
    def train_epoch(self) -> Tuple[float, float]:
        # 훈련모드 설정
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            if self.mixup_fn: # CutMix & Mixup 적용 시
                images, targets = self.mixup_fn(images, targets)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)

            if self.mixup_fn: # cutmix & mixup 적용 시 정확도
                correct += (predicted == targets.argmax(dim=1)).sum().item()
            else:
                correct += (predicted == targets).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(self.train_loader), accuracy

    # 모델의 검증 진행
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                progress_bar.set_postfix(loss=loss.item())
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy

    # 전체 학습 과정 관리
    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            print(f"[Epoch {epoch+1}]\nTrain Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            self.save_model(epoch, val_loss)
            self.scheduler.step()
        
        print("Training finished")