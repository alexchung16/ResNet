# ResNet
ResNet50 & ResNet101

## dataset

https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

## image_preprocessing

https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py

## pre train model

https://github.com/alexchungio/models/tree/master/research/slim

## ResNet50 
### hyper parameter config

* batch size: 32
* learning rate: 0.01
* decay rate: 0.1
* num epoch percent decay: 20
* weight decay: 0.0001
* epoch: 60

| |loss| accuracy
---|---|---
train| 1.2044485807418823|96875
val| 1.2278560400009155|0.936141312122345



