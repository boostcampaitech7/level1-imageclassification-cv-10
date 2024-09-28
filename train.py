import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import pandas as pd

from src.args import get_args
from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.model import ModelSelector
from src.loss import Loss
from src.trainer import Trainer

def main():
    # 설정
    args = get_args()
    device = torch.device(args.device) # 학습에 사용할 장비(cpu or gpu)

    # 데이터 로드
    train_info = pd.read_csv(args.traindata_info_file)
    num_classes = args.num_classes 

    # 학습 및 검증 데이터 분리
    train_df, val_df = train_test_split(
        train_info, 
        test_size=args.val_ratio,
        stratify=train_info['target']
    )

    # Transform 설정
    transform_selector = TransformSelector("albumentations")
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # Dataset 및 DataLoader 설정
    train_dataset = CustomDataset(args.traindata_dir, train_df, train_transform)
    val_dataset = CustomDataset(args.traindata_dir, val_df, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 설정
    model_selector = ModelSelector(args.model_type, num_classes, args.model_name, args.pretrained)
    model = model_selector.get_model().to(device)

    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-6)

    # Loss 함수 설정
    loss_fn = Loss(num_classes=num_classes, label_smoothing=args.label_smoothing)

    # Mixup & CutMix 값 설정
    mixup_args = {
        'mixup_alpha': 0.5, # mixup 강도
        'cutmix_alpha': 1.0, # cutmix 강도
        'cutmix_minmax': None, # cutmix 영역 최소/최대 비율
        'prob': 0.7, # 배치에 적용될 확률
        'switch_prob': 0.5, # mixup/cutmix 선택 확률, 0.3이면 cutmix 30%/mixup 70%
        'mode': 'batch', # 증강 적용할 단위
        'label_smoothing': 0.1, # 레이블 스무딩 강도
        'num_classes': num_classes
    }

    # Trainer 설정 및 학습
    trainer = Trainer(
        args=args, 
        model=model, 
        model_name=args.model_name,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        result_path=args.model_dir,
        mixup_args=mixup_args
    )
    trainer.train()

    print("Training completed. Best model saved at:", args.model_dir)

if __name__ == "__main__":
    main()