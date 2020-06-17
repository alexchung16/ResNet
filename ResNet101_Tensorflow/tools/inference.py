#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/9 上午11:34
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image as pil_image
# from DataProcess.vgg_preprocessing import preprocess_image

meta_path = os.path.join('../outputs/model', 'model.ckpt-5460.meta')
model_path = os.path.join('../outputs/model', 'model.ckpt-5460')
model_pb_path = os.path.join('../outputs/model', 'model.pb')

image_path = './demo/roses_0.jpg'

class_name = ['daisy','dandelion', 'roses', 'sunflowers', 'tulips']

def image_preprocess(img_path, target_size=(299, 299), color_mode='rgb'):
    """

    :param img_path:
    :param target_size:
    :param img_type:
    :return:
    """
    img = pil_image.open(img_path)

    # convert channel
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    if color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # resize
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        img = img.resize(width_height_tuple, resample=pil_image.NEAREST)

    # convert tensor to array
    img = np.asarray(img, dtype=np.float32)

    # convert image scale to (-1, 1)
    scale = 1 / 128.
    img = np.multiply(img, scale) - 1

    # expand dimension
    img_batch = np.expand_dims(img, axis=0)

    return img_batch


def visualize_predict(predict, class_name):
    """
    visualize predict result
    :param predict:
    :param class_name:
    :return:
    """
    fig, ax = plt.subplots()
    y_pos = np.arange(len(predict))

    ax.barh(y_pos, predict, align='center')
    ax.set_yticks(y_pos)
    ax.set_ylabel('category')
    ax.set_yticklabels(class_name)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('probability')
    ax.set_title('predict result')
    plt.show()


def inference_with_ckpt(model_path, image_path, target_size=(299, 299)):
    """

    :param model_path:
    :param image_path:
    :return:
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        restorer = tf.train.import_meta_graph(meta_path)

        # get graph
        graph = tf.get_default_graph()
        # restorer.restore(sess, save_path=tf.train.latest_checkpoint(checkpoint_dir='./outputs/model'))
        restorer.restore(sess, save_path=model_path)

        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        image_placeholder = graph.get_tensor_by_name('input_images:0')
        # label_placeholder = graph.get_tensor_by_name('class_label:0')
        # logits_op = graph.get_tensor_by_name(name='vgg16/prob:0')
        logits_op = graph.get_tensor_by_name('resnet_v2_101/Predict/Softmax:0')

        image_batch = image_preprocess(image_path, target_size=target_size)

        prob = sess.run(logits_op, feed_dict={image_placeholder: image_batch})

        return prob


def predict_with_pb(model_path, image_path, target_size=(299, 299)):
    """
    model read and predict
    :param model_path:
    :param image_data:
    :param input_op_name:
    :param logits_op_name:
    :return:
    """

    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_path, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # get graph
            graph = tf.get_default_graph()
            # get tensor name
            # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
            input_op = graph.get_tensor_by_name(name='input_images:0')
            # logits_op = graph.get_tensor_by_name(name='vgg16/prob:0')
            logits_op = graph.get_tensor_by_name(name='resnet_v2_101/Predict/Softmax:0')
            image_batch = image_preprocess(image_path, target_size=target_size)
            prob = sess.run(fetches=logits_op, feed_dict={input_op: image_batch})

            return prob

if __name__ == "__main__":

        # prob = inference_with_ckpt(model_path, image_path)
        prob = predict_with_pb(model_path=model_pb_path, image_path=image_path)
        predict_label = int(np.argmax(prob))
        print('This is a {0} with possibility {1}'.format(class_name[predict_label], prob[0][predict_label]))

        visualize_predict(prob[0], class_name)