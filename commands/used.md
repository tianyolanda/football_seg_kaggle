# 运行
## 
source activate py37
cd Desktop/kaggle/keras-seg/image-segmentation-keras-master/

# 预测
## 注：checkpoints的全名是：vgg_unet_1.00001.data-00000-of-00001
## 则 "path_to_checkpoints" 写： vgg_unet_1
## 注2：下面的命令需要在image-segmentation-keras-master/下运行，并使用相对路径！！！！

python -m keras_segmentation predict \
 --checkpoints_path="path_to_checkpoints" \
 --input_path="dataset1/images_prepped_test/" \
 --output_path="path_to_predictions"

# 自己finetune的

 python -m keras_segmentation predict \
 --checkpoints_path="checkpoints/dataset2_ori/pspnet_101_finetune_dataset2/pspnet_101_finetune_dataset2" \
 --input_path="dataset2/val_img/" \
 --output_path="color_test/"
 
# 自己训练的vgg_unet_1 测试  \

 python -m keras_segmentation predict_maps \
 --checkpoints_path="checkpoints/dataset2_ori/vgg_unet_300_300/vgg_unet_1" \
 --input_path="dataset2/val_img/" \
 --output_path="color_test2/aaa"

# kaggle

 python -m keras_segmentation predict \
 --checkpoints_path="checkpoints/from_kaggle/vgg_unet" \
 --input_path="dataset2/val_img/" \
 --output_path="color_test/ \
 --prediction_width=1080 \
 --prediction_height=1920"


# 自己训练的fcn_8 测试
 python -m keras_segmentation predict \
 --checkpoints_path="checkpoints/dataset2_ori/fcn_8_416_608/fcn_8_416_608_1" \
 --input_path="dataset2/val_img/" \
 --output_path="dataset2/"

 # 打开tensorboard
 tensorboard --logdir=logs/
 http://localhost:6006/
