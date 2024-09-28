import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Sketch Image Classification')
    
    # 데이터 관련 파라미터
    parser.add_argument('--traindata_dir', type=str, default='./data/train', help='Path to training data directory') # 힉습 데이터 경로
    parser.add_argument('--traindata_info_file', type=str, default='./data/train.csv', help='Path to training data info file') # 학습 데이터 정보 경로
    parser.add_argument('--testdata_dir', type=str, default='./data/test', help='Path to test data directory') # 검증 데이터 경로
    parser.add_argument('--testdata_info_file', type=str, default='./data/test.csv', help='Path to test data info file') # 검증 데이터 정보 경로
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set size (default: 0.2)') # 검증 데이터 비율 설정
    
    # 모델 관련 파라미터
    parser.add_argument('--model_type', type=str, default='timm', help='Model type (default: timm)') # 모델 타입
    parser.add_argument('--model_name', type=str, default='swin_s3_base_224', help='Model name (default: swin_s3_base_224)') # 모델 이름
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model (default: True)') # 사전 학습된 모델 사용 여부
    parser.add_argument('--num_classes', type=int, default=500, help='Number of classes (default: 500)') # 클래스 수

    # 학습 관련 파라미터
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)") # 사용할 디바이스(gpu or cpu)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)') # 배치 크기
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)') # 데이터 로드 시 사용할 워커 수

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 30)') # 에폭 수

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)') # 학습률
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (default: 0.01)') # 가중치 감쇠 파라미터
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (default: 0.0)') # 라벨 스무딩 파라미터
    
    # Other parameters
    parser.add_argument("--model_dir", type=str, default="./train_result",help="Path to save results") # 모델 저장 경로 
    parser.add_argument('--output_dir', type=str, default='./output', help='path to save predictions') # 예측 결과(csv) 저장 경로
    
    return parser.parse_args()
