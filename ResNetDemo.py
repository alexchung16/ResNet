#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File ResNetDemo.py
# @ Description :
# @ Author alexchung
# @ Time 21/1/2019 09:52

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2



images = tf.Variable(initial_value=tf.random_uniform(shape=(5, 299, 299, 3), minval=0, maxval=3), dtype=tf.float32)
num_classes = tf.constant(value=5, dtype=tf.int32)
# is_training = True

if __name__ == "__main__":

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
        # images, class_num = sess.run([images, class_num])
        sess.run(init)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(images, num_classes=num_classes.eval(), is_training=True)

        for var in tf.model_variables():
            print(var.name)
