# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:42.450520Z","iopub.execute_input":"2023-01-04T11:53:42.450888Z","iopub.status.idle":"2023-01-04T11:53:42.459292Z","shell.execute_reply.started":"2023-01-04T11:53:42.450857Z","shell.execute_reply":"2023-01-04T11:53:42.458279Z"}}
# from keras_segmentation.models.fcn import fcn_32
# from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.models.unet import mobilenet_unet
# from keras_segmentation.models.segnet import segnet

from keras.callbacks import Callback, ModelCheckpoint
from imgaug import augmenters as iaa

import os
import time
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np
import glob
import cv2


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:42.463448Z","iopub.execute_input":"2023-01-04T11:53:42.463906Z","iopub.status.idle":"2023-01-04T11:53:42.477821Z","shell.execute_reply.started":"2023-01-04T11:53:42.463876Z","shell.execute_reply":"2023-01-04T11:53:42.476887Z"}}
class TrainingPlot(Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs=None):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_train_end(self, logs=None):
        # only in train end, draw log figure
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")
            colors = ['orange', 'purple', 'green', 'red']
            plt.gca().set_prop_cycle(color=colors)

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.ylim(0, 1)
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_accuracy")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_accuracy")
            # plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.title("Training Loss and Accuracy")

            plt.xlabel("Epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('000_log.png')
            plt.show()
            plt.close()

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        # print('logs:',logs)

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)

        a = logs.get('loss')
        b = logs.get('val_loss')
        if a > 1:
            a = 0.9999
        if b > 1:
            b = 0.9999
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:42.482542Z","iopub.execute_input":"2023-01-04T11:53:42.483479Z","iopub.status.idle":"2023-01-04T11:53:42.492291Z","shell.execute_reply.started":"2023-01-04T11:53:42.483442Z","shell.execute_reply":"2023-01-04T11:53:42.491241Z"}}
trainplot = TrainingPlot()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:42.510393Z","iopub.execute_input":"2023-01-04T11:53:42.510739Z","iopub.status.idle":"2023-01-04T11:53:42.515730Z","shell.execute_reply.started":"2023-01-04T11:53:42.510709Z","shell.execute_reply":"2023-01-04T11:53:42.514700Z"}}
# def custom_augmentation():
#     return iaa.SomeOf((0,None), [
#             # apply the following augmenters to most images
#             iaa.Affine(shear=(-5, 5), mode="reflect"), # symmetric,reflect
#             iaa.Affine(rotate=(-5, 5), mode="reflect"),
#             iaa.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)}),
#             iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#         ])

#     return  iaa.Sequential(
#         [
# #             iaa.Affine(shear=(-5, 5), mode="reflect"),
# #             iaa.Affine(rotate=(-5, 5), mode="reflect"),# constant, wrap
#             iaa.Affine(scale={"x": (1, 1.5), "y": (1, 1.5)}),
# #             iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#         ])


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:42.517964Z","iopub.execute_input":"2023-01-04T11:53:42.518792Z","iopub.status.idle":"2023-01-04T11:53:43.153187Z","shell.execute_reply.started":"2023-01-04T11:53:42.518754Z","shell.execute_reply":"2023-01-04T11:53:43.152215Z"}}
# start processing......
model = mobilenet_unet(n_classes=11, input_height=416, input_width=608)
checkpoint_file = "000_weight.h5"
# cbs = [ModelCheckpoint(checkpoint_file, verbose=0, save_best_only=True, moniter='val_loss'),
#        trainplot,]
cbs = [ModelCheckpoint(checkpoint_file, verbose=1, save_best_only=True),
       trainplot, ]

# %% [code] {"execution":{"iopub.status.busy":"2023-01-04T11:53:43.155322Z","iopub.execute_input":"2023-01-04T11:53:43.155690Z","iopub.status.idle":"2023-01-04T11:53:43.160957Z","shell.execute_reply.started":"2023-01-04T11:53:43.155654Z","shell.execute_reply":"2023-01-04T11:53:43.159971Z"}}
lc_weights = np.array([1, 5.7, 5, 2.8, 2.9, 6.1, 7.5, 4.7, 1.5, 2, 1])  # log(x)后归一化

# lc_weights = np.array([1, 1, 1, 1, 1, 1, 10, 1, 0.5, 0.5, 0.5]) # 单独ball突出
# lc_weights = np.array([1, 8, 4, 2, 2, 8, 16, 4, 1, 1, 1]) # 指数型
# lc_weights = np.array([1, 4, 3, 2, 2, 4, 5, 3, 1, 1, 1]) # 线性
# l_gamma = np.array([1, 1, 1, 1, 1, 1, 10, 1, 0.5, 0.5, 0.5])
# l_fc = [lc_weights,l_gamma]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T11:53:43.162579Z","iopub.execute_input":"2023-01-04T11:53:43.163229Z","iopub.status.idle":"2023-01-04T12:09:15.424112Z","shell.execute_reply.started":"2023-01-04T11:53:43.163193Z","shell.execute_reply":"2023-01-04T12:09:15.423112Z"}}
model.train(
    train_images="/kaggle/input/football-seg-v3/train_img/",
    train_annotations="/kaggle/input/football-seg-v3/train_label/",
    # checkpoints_path = "/kaggle/input/football-seg-v3/checkpoints/pspnet_101_finetune_dataset2/pspnet_101_finetune_dataset2" ,
    loss_fun='weighted_categorical_crossentropy',  # 'multiclass_focal_loss',
    loss_class_weights=lc_weights,  # l_fc,
    epochs=60,
    validate=True,
    val_images='/kaggle/input/football-seg-v3/val_img/',
    val_annotations='/kaggle/input/football-seg-v3/val_label/',
    batch_size=2,
    steps_per_epoch=40,  # 40
    val_steps_per_epoch=40,  # 40
    callbacks=cbs,
    #     do_augment=True, # enable augmentation
    #     custom_augmentation=custom_augmentation # sets the augmention function to use
)

# %% [markdown]
# Now training is finish, and the best weight is saved.
#
# We need to read the best weight, and make evaluation.

# %% [code] {"execution":{"iopub.status.busy":"2023-01-04T12:09:15.427196Z","iopub.execute_input":"2023-01-04T12:09:15.427578Z","iopub.status.idle":"2023-01-04T12:09:15.434411Z","shell.execute_reply.started":"2023-01-04T12:09:15.427541Z","shell.execute_reply":"2023-01-04T12:09:15.433384Z"}}
import json

config_file = "000_config.json"
n_classes = model.n_classes
input_height = model.input_height
input_width = model.input_width
output_height = model.output_height
output_width = model.output_width
with open(config_file, "w") as f:
    json.dump({
        "model_class": model.model_name,
        "n_classes": n_classes,
        "input_height": input_height,
        "input_width": input_width,
        "output_height": output_height,
        "output_width": output_width
    }, f)

# %% [code] {"execution":{"iopub.status.busy":"2023-01-04T12:09:15.436154Z","iopub.execute_input":"2023-01-04T12:09:15.436514Z","iopub.status.idle":"2023-01-04T12:09:15.448462Z","shell.execute_reply.started":"2023-01-04T12:09:15.436475Z","shell.execute_reply":"2023-01-04T12:09:15.447548Z"}}
n_classes, input_height, input_width, output_height, output_width

# %% [code] {"execution":{"iopub.status.busy":"2023-01-04T12:09:15.449894Z","iopub.execute_input":"2023-01-04T12:09:15.450302Z","iopub.status.idle":"2023-01-04T12:09:16.350017Z","shell.execute_reply.started":"2023-01-04T12:09:15.450269Z","shell.execute_reply":"2023-01-04T12:09:16.349098Z"}}
from keras_segmentation.predict import model_from_checkpoint_path_v2

model_best = model_from_checkpoint_path_v2(config_file, checkpoint_file)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:16.351396Z","iopub.execute_input":"2023-01-04T12:09:16.351744Z","iopub.status.idle":"2023-01-04T12:09:16.364493Z","shell.execute_reply.started":"2023-01-04T12:09:16.351709Z","shell.execute_reply":"2023-01-04T12:09:16.363025Z"}}
def show_multi_imgs(scale, imglist, order=None, border=10, border_color=(255, 255, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws, hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h + border) * order[0], (w + border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i * border):((i + 1) * h + i * border), (j * w + j * border):((j + 1) * w + j * border),
            :] = allimgs[i * order[1] + j]
    return imgblank


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:16.366213Z","iopub.execute_input":"2023-01-04T12:09:16.366562Z","iopub.status.idle":"2023-01-04T12:09:19.309145Z","shell.execute_reply.started":"2023-01-04T12:09:16.366527Z","shell.execute_reply":"2023-01-04T12:09:19.308119Z"}}
val_img_path = '/kaggle/input/football-seg-v3/val_img/*'
val_img_list = glob.glob(val_img_path)
val_img_list.sort()

val_result = []
for i in range(len(val_img_list)):
    out = model_best.predict_maps(inp=val_img_list[i])
    # print('out.shape',out.shape)
    # cv2.imshow(str(i), out)
    # cv2.waitKey(0)
    #     out = out[:, :, (2, 1, 0)] # cv2的BGR->plt的RGB
    val_result.append(out)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:19.310724Z","iopub.execute_input":"2023-01-04T12:09:19.311172Z","iopub.status.idle":"2023-01-04T12:09:21.384351Z","shell.execute_reply.started":"2023-01-04T12:09:19.311134Z","shell.execute_reply":"2023-01-04T12:09:21.383382Z"}}
# show_and_save_maps(val_result)
img = show_multi_imgs(0.5, val_result, (6, 3))
plt.imshow(img[:, :, (2, 1, 0)])
# cv2.namedWindow('val_result_multi', 0)
# cv2.imshow('val_result_multi', img)

cv2.imwrite('000_val_result_multi.png', img)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:21.388013Z","iopub.execute_input":"2023-01-04T12:09:21.388599Z","iopub.status.idle":"2023-01-04T12:09:24.190106Z","shell.execute_reply.started":"2023-01-04T12:09:21.388560Z","shell.execute_reply":"2023-01-04T12:09:24.189118Z"}}
# evaluate IoU, fIoU...
eval_result = model_best.evaluate_segmentation(inp_images_dir="/kaggle/input/football-seg-v3/val_img/",
                                               annotations_dir="/kaggle/input/football-seg-v3/val_label/")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.191825Z","iopub.execute_input":"2023-01-04T12:09:24.192522Z","iopub.status.idle":"2023-01-04T12:09:24.201018Z","shell.execute_reply.started":"2023-01-04T12:09:24.192483Z","shell.execute_reply":"2023-01-04T12:09:24.200001Z"}}
eval_result


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.204183Z","iopub.execute_input":"2023-01-04T12:09:24.204474Z","iopub.status.idle":"2023-01-04T12:09:24.212901Z","shell.execute_reply.started":"2023-01-04T12:09:24.204444Z","shell.execute_reply":"2023-01-04T12:09:24.211762Z"}}
def number2string_format(inp):
    # 保留4位小数点（四舍五入）：round(l[i], 4)
    # 小数点未满4位的填补0："{:.4f}".format()
    return str("{:.4f}".format(round(inp, 4)))


def list2string(lis):
    # 输入一个list，输出一行string，list的每个数字保留4位小数，数字之间用空格隔开
    s = ''
    for i in range(len(lis)):
        s += number2string_format(lis[i])
        s += ' '
    return s


def write2txt(loss, eval_result, filename):
    file = open(filename, "w")
    i = 0

    # 先把eval_result写入
    for value in eval_result.values():
        if i < 2:  # frequency_weighted_IU, mean_IU
            value_format = number2string_format(value)
        else:  # class_wise_IU, class_wise_pixels_norm
            value_format = list2string(value)
        file.write(value_format)
        file.write('\n')
        i += 1

    # 再把loss写入
    for value in loss.values():
        value_format = list2string(value)
        file.write(value_format)
        file.write('\n')

    file.close()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.214530Z","iopub.execute_input":"2023-01-04T12:09:24.215039Z","iopub.status.idle":"2023-01-04T12:09:24.224068Z","shell.execute_reply.started":"2023-01-04T12:09:24.215005Z","shell.execute_reply":"2023-01-04T12:09:24.223120Z"}}
loss_dict = {'train_losses': trainplot.losses,
             'train_acc': trainplot.acc,
             'val_losses': trainplot.val_losses,
             'val_acc': trainplot.val_acc}

# save to txt
write2txt(loss_dict, eval_result, "000_eval.txt")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.225369Z","iopub.execute_input":"2023-01-04T12:09:24.226175Z","iopub.status.idle":"2023-01-04T12:09:24.234419Z","shell.execute_reply.started":"2023-01-04T12:09:24.226138Z","shell.execute_reply":"2023-01-04T12:09:24.233312Z"}}
from IPython.display import FileLink

FileLink('000_weight.h5')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.236085Z","iopub.execute_input":"2023-01-04T12:09:24.236452Z","iopub.status.idle":"2023-01-04T12:09:24.244832Z","shell.execute_reply.started":"2023-01-04T12:09:24.236419Z","shell.execute_reply":"2023-01-04T12:09:24.243683Z"}}
FileLink('000_eval.txt')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.246302Z","iopub.execute_input":"2023-01-04T12:09:24.246783Z","iopub.status.idle":"2023-01-04T12:09:24.254745Z","shell.execute_reply.started":"2023-01-04T12:09:24.246748Z","shell.execute_reply":"2023-01-04T12:09:24.253649Z"}}
FileLink('000_log.png')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-01-04T12:09:24.256091Z","iopub.execute_input":"2023-01-04T12:09:24.257125Z","iopub.status.idle":"2023-01-04T12:09:24.264895Z","shell.execute_reply.started":"2023-01-04T12:09:24.257060Z","shell.execute_reply":"2023-01-04T12:09:24.263867Z"}}
FileLink('000_val_result_multi.png')