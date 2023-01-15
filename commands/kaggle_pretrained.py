from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_101_cityscapes
from keras_segmentation.models.pspnet import pspnet_101
from keras.callbacks import Callback, ModelCheckpoint

import os
import time
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np

class TrainingPlot(Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs=None):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

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
        print(self.acc,self.val_acc)

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")
            colors=['orange', 'purple', 'green','red']
            plt.gca().set_prop_cycle(color=colors)
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.ylim(0, 1) 
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_accuracy")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            #plt.show()

            plt.savefig('log.png')
            plt.close()

pretrained_model = pspnet_101_cityscapes()

new_model = pspnet_101( n_classes=11 ) # original "n_classes": 19 

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

cbs = [ModelCheckpoint("pspnet_101_Segmentation.h5", save_best_only=True), TrainingPlot()]

new_model.train(
    train_images =  "/kaggle/input/football-seg-v3/train_img/",
    train_annotations = "/kaggle/input/football-seg-v3/train_label/",
    #checkpoints_path = "/kaggle/input/football-seg-v3/checkpoints/pspnet_101_finetune_dataset2/pspnet_101_finetune_dataset2" ,
    epochs=80,
    validate=True,
    val_images='/kaggle/input/football-seg-v3/val_img/',
    val_annotations='/kaggle/input/football-seg-v3/val_label/',
    batch_size = 2,
    steps_per_epoch = 40, # 40
    val_steps_per_epoch = 40, # 40
    callbacks = cbs,
)

print(new_model.evaluate_segmentation( inp_images_dir="/kaggle/input/football-seg-v3/val_img/"  , annotations_dir="/kaggle/input/football-seg-v3/val_label/" ) )