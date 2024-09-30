import timm
import torch
import torch.nn as nn

class ModelSelector:
    def __init__(self, model_type, num_classes, model_name, pretrained):
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained

    def get_model(self):
        if self.model_type == 'timm':
            # timm library를 사용하여 모델 생성
            model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model