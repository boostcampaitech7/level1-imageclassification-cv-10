python train.py --model_name swin_s3_base_224 --epochs 20 --lr 0.0001 --weight_decay 0.001 --label_smoothing 0.1

python train.py --model_name coatnet_2_rw_224 --epochs 50 --lr 0.00005 --weight_decay 0.00000001 --label_smoothing 0.1

python train.py --model_name convnext_base --epochs 30 --lr 0.00015 --weight_decay 0.0000005 --label_smoothing 0.0

python train.py --model_name deit_base_patch16_224 --epochs 30 --lr 0.00015 --weight_decay 0.0005 --label_smoothing 0.0

python train.py --model_name tf_efficientnetv2_s --epochs 30 --lr 0.0005 --weight_decay 0.00001 --label_smoothing 0.0