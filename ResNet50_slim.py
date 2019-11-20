#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File ResNet50_slim.py
# @ Description :
# @ Author alexchung
# @ Time 15/11/2019 AM 09:17


import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ResNet50():
    """
    Inception v1
    """

    def __init__(self, input_shape, num_classes, batch_size, num_samples_per_epoch, num_epoch_per_decay,
                 decay_rate, learning_rate, keep_prob=0.8, regular_weight_decay=0.00004, batch_norm_decay=0.9997,
                 batch_norm_epsilon=0.001, batch_norm_scale=False, batch_norm_fused=True, reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.reuse = reuse
        self.regular_weight_decay = regular_weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.batch_norm_fused = batch_norm_fused
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder(tf.float32,
                                                       shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                                       name="input_images")
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits = self.inference(self.raw_input_data, scope='InceptionV4')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.train_accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size

    def inference(self, inputs, scope='InceptionV4'):
        """
        Inception V4 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        pass

    def resnet50(self, inputs, scope='resnet_v2_50', num_classes=10, keep_prob=0.8,
                     reuse=None, is_training=False):
        """
        inception v4
        :return:
        """
        pass

    def resnet50_base(self, inputs, scope='resnet_v2_50'):
        """
        inception V4 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='resnet_v2_50', values=[inputs]):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                pass



    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
        """
        convolution with 'SAME' padding
        :param inputs:
        :param num_outputs:
        :param kernel_size:
        :param stride:
        :param rate:
        :param scope:
        :return:
        """
        if stride == 1:
            return slim.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size,
                               stride=stride, padding='SAME', scope=scope)
        else:
            kernel_size_effect = kernel_size[0] * (kernel_size[0] - 1) * (rate - 1)
            pad_total = kernel_size_effect - 1
            pad_begin = pad_total // 2
            pad_end = pad_total - pad_begin
            inputs = tf.pad(inputs, paddings=[[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
            return slim.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size,
                               stride=stride, padding='VALID', scope=scope)



