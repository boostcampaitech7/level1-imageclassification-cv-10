import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.Dataset import CustomDataset, download_file_unzip
from transform.Transforms import TransformSelector
from model.Model import ModelSelector
from trainer.Trainer import Trainer
from utils.utils import accuracy, Loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    # data download, 건들 필요 없음
    url = 'https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz'
    output_path = 'data.tar.gz'

    download_file_unzip(url, output_path)

    # cuda device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_result"

    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique()) # 총 클래스 수

    # 학습, 검증 데이터 분리
    train_df, val_df = train_test_split(
    train_info,
    test_size=0.2,
    stratify=train_info['target']
    )

    transform_type = "albumentations" # tranform에 사용할 모듈 설정
    MODEL_NAME = 'resnet18' # 모델 이름 설정
    BATCH_SIZE = 64 # 배치 사이즈 설정
    NUM_WORKERS = os.cpu_count() # num_workers 설정

    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    steps_per_epoch = len(train_loader)
    epochs_per_lr_decay = 2
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    # 손실 함수 설정
    loss_fn = Loss()

    transform_selector = TransformSelector(
    transform_type = transform_type
    )
    # Augmentation은 여기서 수정하면 된다.

    # train data 전처리 코드
    train_transforms = A.Compose([
        A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
        A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
        A.Rotate(limit=15),  # 최대 15도 회전
        A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
        ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
    ])
    train_transform = transform_selector.get_transform(transforms= train_transforms)

    # val data 전처리 코드
    val_transforms = A.Compose([
        A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
        ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
    ])
    val_transform = transform_selector.get_transform(transforms= val_transforms)

    # 데이터셋 및 데이터로더
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 모델 설정
    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name=MODEL_NAME, 
        pretrained=True
    )

    model = model_selector.get_model()
    model.to(device)

    # 트레이너 설정
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=5,
        result_path=save_result_path
    )

    # 학습 시작
    trainer.train()

if __name__ == '__main__':
    main()