from keras_segmentation.pretrained import pspnet_101_cityscapes

model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset


out = model.predict_segmentation(
    inp="cam6_eletric_0005.jpg",
    out_fname="out.png"
)