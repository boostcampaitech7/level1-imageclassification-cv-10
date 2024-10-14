import pandas as pd
from pathlib import Path

# CSV 파일 경로를 리스트로 정의합니다.
csv_files = [
    'output (1).csv',     # greedysoup_swin 0.8820
    'output (2).csv',      # pre_trained_boosting(conatNet, swin s3) 0.9020
    'output (3).csv',      # EfficientNetV2 + CoAtNet Boosting 0.8850
    'output (4).csv',      # convnext  0.8850
    'output (5).csv',      # 추측 후 예외처리 0.8950
    'output (6).csv',      # swim + 오분류 레이블 개선 0.8780
    'output (7).csv',     #  Epoch 수 증가 (이재훈) 0.8790
    'output (8).csv',      # base(swin+전처리)+cutmix+mixup(20) (문채원) 0.8780
    'output (9).csv',      #  가중치조정 (문채원) 0.8770
    'output (10).csv',      # coatnet_2_rw_224_전처리(종+그+C)+증강_MixUp (이재훈) 0.8760
    'output (11).csv',      # cutmix+mixup 파라미터 조정한거(20) (문채원) 0.8750
    'output (12).csv',      # convnextv2_tiny (문채원) 0.8500
    'output (13).csv'        # 전처리(종+그+C)_증강(베+가노)_swin_AdamW_CosineAnnealingLR (장지우) 0.8720
]

# 입력 및 출력 디렉토리 설정
input_dir = Path('D:/Downloads')
output_dir = Path('D:/Downloads')

# 각 CSV 파일을 읽어서 'target' 열만 추출하여 DataFrame 리스트로 변환합니다.
targets = [pd.read_csv(input_dir / file)['target'] for file in csv_files]

# DataFrame으로 변환하고 최빈값 계산
df_targets = pd.DataFrame(targets).T
result = df_targets.mode(axis=1).iloc[:, 0]  # 첫 번째 열만 선택 (동점이 있을 경우)

# 원본 CSV 파일 읽기 (첫 번째 파일 사용)
df = pd.read_csv(input_dir / csv_files[0])

# 'target' 열을 새로운 결과로 대체
df['target'] = result

# 변경된 내용을 새로운 CSV 파일로 저장
output_file = output_dir / 'Voting_Output.csv'
df.to_csv(output_file, index=False)

print(f"Hard voting completed. Results saved to {output_file}")