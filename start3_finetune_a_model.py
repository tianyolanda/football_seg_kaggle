from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_101_cityscapes
from keras_segmentation.models.pspnet import pspnet_101

pretrained_model = pspnet_101_cityscapes()

new_model = pspnet_101( n_classes=11 ) # original "n_classes": 19 

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "dataset2/train_img/",
    train_annotations = "dataset2/train_label/",
    checkpoints_path = "checkpoints/pspnet_101_finetune_dataset2" , epochs=5
)

# 如果能运行了，加一下validation部分再开始训