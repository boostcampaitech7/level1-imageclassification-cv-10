import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Callable

class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        if is_train:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ])

    def __call__(self, image):
        return self.transform(image=image)['image']

class AlbumentationsTransform:
    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed['image']  # 변환된 이미지의 텐서를 반환

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type

        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, transforms):

        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)

        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(transform= transforms)

        return transform