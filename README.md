# ResNet
ResNet50 & ResNet101

## dataset

https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

## image_preprocessing
> The key difference of the full preactivation 'v2' variant compared to the 'v1' variant in [1] is the use of batch normalization before every weight layer.
* resnet(V1)
** ResNet V1 models use vgg pre-processing and input image size of 224*224 **
<https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py>
* resnet(V2)
according to the <https://github.com/alexchungio/models/tree/master/research/slim>
** ResNet V2 models use Inception pre-processing and input image size of 299*299 **
<https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py>
## pre train model

https://github.com/alexchungio/models/tree/master/research/slim

## ResNet_v2_50 
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


## ResNet_v2_101 
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

