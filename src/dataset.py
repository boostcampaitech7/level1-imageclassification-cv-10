import os
from typing import Tuple, Callable, Union

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋 초기화
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        # 학습 시, 각 이미지에 대한 타겟 정보 저장 
        self.targets = info_df['target'].tolist() if not is_inference else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        # BGR -> RGB 변환
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 이미지 변환 적용
        transformed = self.transform(image=image)
        image = transformed["image"]

        if self.is_inference:
            return image # 추론 시, 이미지만 반환
        else:
            return image, self.targets[index] # 학습 시 ,이미지와 타겟 반환