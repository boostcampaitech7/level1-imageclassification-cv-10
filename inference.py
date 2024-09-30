import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.model import ModelSelector
from src.args import get_args

def load_data(file_path: str, is_test: bool = True):
    df = pd.read_csv(file_path)
    return df

def inference(model, device, test_loader):
    model.eval() # 평가모드로 설정
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().detach().numpy())
    return predictions

def main():
    args = get_args()

    device = torch.device(args.device)

    # 테스트 데이터 로드
    test_info = load_data(args.testdata_info_file, is_test=True)

    # Transform 설정
    transform_selector = TransformSelector("albumentations")
    test_transform = transform_selector.get_transform(is_train=False)

    # Dataset 및 DataLoader 설정
    test_dataset = CustomDataset(args.testdata_dir, test_info, test_transform, is_inference=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # 모델 로드
    model_selector = ModelSelector(args.num_classes, model_name=args.model_name, pretrained=args.pretrained)
    model = model_selector.get_model().to(device)
    model_path = os.path.join(args.model_dir, f"{args.model_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 추론
    predictions = inference(model, device, test_loader)

    # 결과 저장
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    
    # 출력 파일 설정
    output_filename = f"{args.model_name}_output.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    os.makedirs(args.output_dir, exist_ok=True)
    test_info.to_csv(output_path, index=False)

    print(f"Inference completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()