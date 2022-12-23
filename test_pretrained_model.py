from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12, resnet_pspnet_VOC12_v0_1

# model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

# model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# model = resnet_pspnet_VOC12_v0_1()

# load any of the 3 pretrained models


name = 'pspnet101_city'
for i in range(10):
    out = model.predict_segmentation(
        inp="dataset2/val_img/0"+str(i)+ "0.png",
        out_fname="dataset2/test_result/"+name+"/"+name+"_out_0"+str(i)+"0.png"
    )
    print(i,'finish')