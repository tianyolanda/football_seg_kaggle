
from keras_segmentation.models.unet import vgg_unet
from imgaug import augmenters as iaa

def custom_augmentation():
    return  iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5), # horizontally flip 50% of all images
        ])

model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608)

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5, 
    do_augment=True, # enable augmentation 
    custom_augmentation=custom_augmentation # sets the augmention function to use
)