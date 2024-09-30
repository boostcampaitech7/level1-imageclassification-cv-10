# hard voting code
import pandas as pd
from scipy import stats

# 5개의 CSV 파일 경로를 리스트로 정의합니다. 
csv_files = ['D:/Downloads/output (1).csv',     # greedysoup_swin 0.8820
            'D:/Downloads/output (2).csv',      # pre_trained_boosting(conatNet, swin s3) 0.9020
            'D:/Downloads/output (3).csv',      # EfficientNetV2 + CoAtNet Boosting 0.8850
            'D:/Downloads/output (4).csv',      # convnext  0.8850
            'D:/Downloads/output (5).csv',      # 추측 후 예외처리 0.8950
            'D:/Downloads/output (6).csv',      # swim + 오분류 레이블 개선 0.8780
            'D:/Downloads/output (7).csv',     #  Epoch 수 증가 (이재훈) 0.8790
            'D:/Downloads/output (8).csv',      # base(swin+전처리)+cutmix+mixup(20) (문채원) 0.8780
            'D:/Downloads/output (9).csv',      #  가중치조정 (문채원) 0.8770
            'D:/Downloads/output (10).csv',      # coatnet_2_rw_224_전처리(종+그+C)+증강_MixUp (이재훈) 0.8760
            'D:/Downloads/output (11).csv',      # cutmix+mixup 파라미터 조정한거(20) (문채원) 0.8750
            'D:/Downloads/output (12).csv',      # convnextv2_tiny (문채원) 0.8500
            'D:/Downloads/output (13).csv']        # 전처리(종+그+C)_증강(베+가노)_swin_AdamW_CosineAnnealingLR (장지우) 0.8720
# 각 CSV 파일을 읽어서 DataFrame 리스트로 변환합니다.
dfs = [pd.read_csv(file) for file in csv_files]
targets = []
for df in dfs:
    targets.append(df['target'])

# 새로운 DataFrame에 각 DataFrame의 'target' 값을 추가
new_df = pd.DataFrame()
for i in range(len(dfs)):
    new_df = pd.concat([new_df, dfs[i]['target']], ignore_index=True, axis=1)

result = []
for idx in range(len(targets[0])):
    target_list = []
    for target in targets:
        target_list.append(target[idx])
    # mode() 함수를 사용해 최빈값을 계산
    target_result = stats.mode(target_list)
    result.append(target_result.mode)


df = pd.read_csv('D:/Downloads/output.csv')

# 기존의 'target' 열을 new_values 배열로 대체합니다.
df['target'] = result

# 필요하다면 변경된 내용을 새로운 CSV 파일로 저장할 수 있습니다.
df.to_csv('D:/Downloads/Voting_Output.csv', index=False)
