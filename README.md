# Sketch Image Classification
## Overview
2024.09.10 ~ 2022.09.26

This project focuses on classifying ImageNet-Sketch images as part of a private competition organized by Naver Connect Foundation and Upstage.


## Contributors
|김기수|문채원|안주형|은의찬|이재훈|장지우
|:----:|:----:|:----:|:----:|:----:|:----:|
| [<img src="https://github.com/user-attachments/assets/366fc4d1-3716-4214-a6ef-87f0a4c6147f" alt="" style="width:100px;100px;">](https://github.com/Bbuterfly) <br/> | [<img src="https://github.com/user-attachments/assets/ea61c11c-c577-45bb-ae8e-64dffa192402" alt="" style="width:100px;100px;">](https://github.com/mooniswan) <br/> | [<img src="https://github.com/user-attachments/assets/6bc5913f-6e59-4aae-9433-3db2c7251978" alt="" style="width:100px;100px;">](https://github.com/Ahn-latte) <br/> | [<img src="https://github.com/user-attachments/assets/22d440d4-516b-4973-a2fe-06adc145fa01" alt="" style="width:100px;100px;">](https://github.com/0522chan) <br/> | [<img src="https://github.com/user-attachments/assets/3ed91d99-0ad0-43ee-bb11-0aefc61a0a0e" alt="" style="width:100px;100px;">](https://github.com/syous154) <br/> | [<img src="https://github.com/user-attachments/assets/04f5faa7-05c4-4ecc-87f1-0befb53da70d" alt="" style="width:100px;100px;">](https://github.com/zangzoo) <br/> |

## Experimental Techniques

We experimented with various techniques to improve performance:

- **Augmentation**: Aspect ratio, CLAHE, grayscale, flip, rotation, mixup, cutmix, Text-to-Image
- **Optimization**: Pseudo labeling, label smoothing, Error-Driven Learning, Progressive resizing, Model Soups, TTA
- **Visualization**: Confusion matrix, error-class visualization
- **Ensemble**: Soft voting, Hard voting, Ensemble of Ensembles, stacking, snapshot


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
├── inference.sh
├── ensemble.py
├── requirements.txt
├── .gitignore
└── README.md
```
- `data/train/`: 15,021 training images across 500 classes (29-31 images per class)
- `data/test/`: 10,014 test images

## Getting Started

### Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```

## Hyperparameters
The following arguments can be adjusted via command line to experiment with different settings:

|argument option|default|Description|
|---|---|---|
|traindata_dir|./data/train|Path to training data dir|
|traindata_info_file|./data/train.csv|Name of the training info file|
|testdata_dir|./data/test|Path to test data dir|
|testdata_info_file|./data/test.csv|Name of the test info file|
|val_ratio|0.2|Validation data ratio|
|model_type|timm|Model type|
|model_name|swin_s3_base_224|Model name|
|pretrained|True|Use pretrained model|
|num_classes|500|Number of classes|
|device|cuda|Device to use(gpu or cpu)|
|batch_size|64|Batch size|
|num_workers|4|Number of workers|
|epochs|10|Number of epochs|
|lr|0.001|Learning rate|
|weight_decay|0.0001|Weight decay coefficient|
|label_smoothing|0.0|Label smoothing factor|
|model_dir|./train_result|Path to save training results (.pt)|
|output_dir|./output|Path to save prediction results (.csv)|


### Training Step
To train a single model, run following command.
```bash
python train.py --model {model_name} --batch_size {batch_size} --lr {learning rate} --epochs {epochs} --label_smoothing {label smoothing}
```

To train multiple models, run following shell script file.
```sh
bash train.sh
```

### Inference Step
To test model, run following command.
```bash
python inference.py --model_name {model_name}
```

#### Ensemble
To perform ensemble prediction:
```bash
python ensemble.py
```
