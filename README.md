# Sketch Image Classification
 
![image](https://github.com/user-attachments/assets/6240dc62-b3f1-40e1-9eb8-25f13f15112b)

- 2024.09.10 ~ 2022.09.26
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- Sketch 이미지를 분류하는 Image Classification task

### Datasets
- 전체 이미지 개수: 25,035장
- 클래스 수: 500개, class별 29~31개의 이미지를 가짐
- ImageNet-Sketch 데이터셋을 사용

![image](https://github.com/user-attachments/assets/62a9e117-f05d-4590-b01e-2cb73ad3bdfd)

## Folder Structure
```
level1-cv-10
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── args.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── loss.py
│   ├── trainer.py
├── train.py
├── train.sh
├── inference.py
├── ensemble.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 실험 과정
모델의 실험과정에서 사용되는 하이퍼파라미터 값들은 arguments로 전달할 수 있다.

- train.py
    |argument option|default|Description|
    |---|---|---|
    |ltraindata_dir|./data/train|힉습 데이터 경로|
    |traindata_info_file|./data/train.csv|학학습 데이터 정보 파일 이름|
    |testdata_dir|./data/test|검증 데이터 경로|
    |testdata_info_file|./data/test.csv|검증 데이터 정보 파일 이름|
    |val_ratio|0.2|검증 데이터 비율|
    |model_type|timm|모델 타입|
    |model_name|swin_s3_base_224|모델 이름|
    |pretrained|True|# 사전 학습된 모델 사용 여부|
    |num_classes|500|분류할 클래스 수|
    |device|cuda|사용할 디바이스(gpu or cpu)|
    |batch_size|64|dataloader에 지정할 배치 크기|
    |num_workers|4|데이터 로드 시 사용할 워커 수|
    |epochs|10|학습 epoch 수|
    |lr|0.001|학습률|
    |weight_decay|0.0001|가중치 감쇠 계수|
    |label_smoothing|0.0|라벨 스무딩 계수|
    |model_dir|./train_result|학습 결과 저장 경로|
    |output_dir|./output|예측 결과(csv) 저장 경로|



## 사용 방법
### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

대회에 제출한 7개의 모델을 한 번에 학습시키길 원한다면, 아래의 쉘 스크립트 파일을 실행하면 된다.
```sh
$ ./train.sh
```

### Inference
```bash
python inference.py
```

여러개의 모델을 한 번에 추론, 앙상블하고 싶다면 아래의 쉘 스크립트 파일을 실행하면 된다.
```sh
$ ./inference.sh
```