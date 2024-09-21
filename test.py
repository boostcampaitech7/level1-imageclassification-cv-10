import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from tqdm.auto import tqdm
from albumentations.pytorch import ToTensorV2
from model.Model import ModelSelector
from dataset.Dataset import CustomDataset
from transform.Transforms import TransformSelector

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"
    save_result_path = "./train_result"

    test_info = pd.read_csv(testdata_info_file)

    # 총 class 수.
    transform_type = "albumentations"
    num_classes = 500
    BATCH_SIZE = 64 # 배치 사이즈 설정
    MODEL_NAME = 'resnet18' # 모델 이름 설정

    # 추론에 사용할 Transform을 선언.
    transform_selector = TransformSelector(
        transform_type = transform_type
    )

    test_transforms = A.Compose([
        A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
        ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
    ]) # 테스트 데이터 증강 설정

    test_transform = transform_selector.get_transform(transforms= test_transforms)
    # 추론에 사용할 Dataset을 선언.
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )
    
    # 추론에 사용할 DataLoader를 선언.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    model_selector = ModelSelector(
    model_type='timm',
    num_classes=num_classes,
    model_name=MODEL_NAME,
    pretrained=False
    )
    model = model_selector.get_model()

    model.load_state_dict(
    torch.load(
        os.path.join(save_result_path, "best_model.pt"),
        map_location='cpu'
        )
    )

    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)

            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    
    # DataFrame 저장
    test_info.to_csv("output.csv", index=False)

if __name__ == '__main__':
    main()