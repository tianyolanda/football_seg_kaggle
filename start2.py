#from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import TensorBoard
from time import time
from keras_segmentation.models.segnet import resnet50_segnet
# from keras_segmentation.models.fcn import fcn_8

import os

# model = vgg_unet(n_classes=11, input_height=320, input_width=320)
# model = fcn_8(n_classes=11, input_height=416, input_width=608)
# model = vgg_unet(n_classes=11, input_height=416, input_width=608)
model = resnet50_segnet(n_classes=11, input_height=416, input_width=608)

# define parameters: model and dataset 
model_name = 'resnet50_segnet_416_608'
session = 1
dataset_name = 'dataset2_ori'
name = model_name + '_' + dataset_name

checkpoints_save_dir = "checkpoints/"+ dataset_name+"/"+model_name+"/"
test_img_save_dir = "dataset2/test_result/"+name+"/"

# create dir : checkpoints and test_image save
if not os.path.exists(checkpoints_save_dir):
    print('create checkpoints dir...',checkpoints_save_dir)    
    os.makedirs(checkpoints_save_dir)
else:
    print('checkpoints dir already exists.',checkpoints_save_dir)

if not os.path.exists(test_img_save_dir):
    print('create test_result dir...',test_img_save_dir)
    os.makedirs(test_img_save_dir)
else:
    print('test_result dir already exists.',test_img_save_dir)

# Create a TensorBoard instance with the path to the logs directory
print('tensorboard start...')
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(time())) 

# train nmodel 
model.train(
    train_images =  "dataset2/train_img/",
    train_annotations = "dataset2/train_label/",
    checkpoints_path = checkpoints_save_dir+model_name+"_"+str(session),
    epochs = 5,
    validate = True,
    val_images = 'dataset2/val_img/',
    val_annotations = 'dataset2/val_label/',
    callbacks = [tensorboard],
)


for i in range(10):
    out = model.predict_segmentation(
        inp="dataset2/val_img/0"+str(i)+ "0.png",
        out_fname= test_img_save_dir+name+"_out_0"+str(i)+"0.png"
    )

print('finish validation, and saved in ',test_img_save_dir)

# import matplotlib.pyplot as plt
# plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset2/val_img/"  , annotations_dir="dataset2/val_label/" ) )