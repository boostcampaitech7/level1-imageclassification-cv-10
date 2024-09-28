import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Transforms:
    @staticmethod
    def get_transform(is_train: bool = True):
        # 공통 변환
        common_transforms = [
            A.LongestMaxSize(max_size=224), # 가장 큰 축에 맞춰 크기 조정
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_REFLECT), # 패딩 추가해 224x224 resize
            A.ToGray(p=1.0), # Greyscale 변환
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # CLAME 적용
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 정규화
            ToTensorV2() # 텐서 변환
        ]
        # 훈련용 변환
        if is_train:
            train_transforms = [
                A.HorizontalFlip(p=0.5), # 수평 flip(50% 확률)
                A.Rotate(limit=15), # 15도 이내 회전
                A.RandomBrightnessContrast(p=0.2), # 밝기 및 대비 조절
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # 가우시안 노이즈
            ]
            return A.Compose(train_transforms + common_transforms)
        # 검증/테스트용 변환
        else:
            return A.Compose(common_transforms)

    @staticmethod
    def __call__(image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        transform = Transforms.get_transform(is_train=True) 
        transformed = transform(image=image)
        return transformed['image']

class TransformSelector:
    def __init__(self, transform_type: str):
        self.transform_type = transform_type

    def get_transform(self, is_train: bool):
        if self.transform_type == "albumentations":
            return Transforms.get_transform(is_train)
