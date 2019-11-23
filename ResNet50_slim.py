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
                 decay_rate, learning_rate, keep_prob=0.8, weight_decay=0.0001, batch_norm_decay=0.997,
                 batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_fused=True, is_pretrain=False,
                 reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.is_pretrain = is_pretrain
        self.reuse = reuse
        self.weight_decay = weight_decay
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

        # self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_step = tf.train.get_or_create_global_step()
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits = self.inference(self.raw_input_data, scope='resnet_v2_50')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step, loss=self.loss)
        self.accuracy = self.evaluate_batch(self.logits, self.raw_input_label) / batch_size

    def inference(self, inputs, scope='resnet_v2_50'):
        """
        Inception V4 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        prop = self.resnet50(inputs=inputs,
                             scope=scope,
                             num_classes=self.num_classes,
                             keep_prob=self.keep_prob,
                             reuse=self.reuse,
                             is_training=self.is_training)

        return prop

    def resnet50(self, inputs, scope='resnet_v2_50', num_classes=10, keep_prob=0.8,
                     reuse=None, is_training=False):
        """
        inception v4
        :return:
        """
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': self.batch_norm_epsilon,
            'scale': self.batch_norm_decay,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }
        with tf.variable_scope( scope, 'resnet_v2_50', [inputs], reuse=reuse) as sc:
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(self.weight_decay),
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.layers.batch_norm,
                    normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        with slim.arg_scope([slim.batch_norm], is_training=is_training):
                            net = self.resnet50_base(inputs=inputs, scope=sc)
                            # batch normalize
                            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                            # average pool
                            net = tf.reduce_mean(input_tensor=net, axis=[1, 2], name='pool5', keep_dims=True)
                            # logits
                            logits = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=[1, 1],
                                                 activation_fn=None, normalizer_fn=None, scope='logits')
                            # squeeze
                            logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')
                            # softmax
                            prop = slim.softmax(logits, scope='predict')
                            return prop

    def resnet50_base(self, inputs, scope='resnet_v2_50'):
        """
        inception V4 base
        :param inputs:
        :param scope:
        :return:
        """
        with tf.compat.v1.variable_scope(scope, default_name='resnet_v2_50', values=[inputs]):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                # 224 x 224 x 3
                net = self.conv2d_same(inputs=inputs, kernel_size=[7, 7], num_outputs=64, stride=2,
                                       scope='conv1')
            # 112 x 112 x 3
            net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope='pool1')
            # block 1
            # 56 x 56 x 3
            with tf.variable_scope('block1', 'block') as sc:
                net = self.conv_block(inputs=net, filters=[64, 64, 256], stride=2, scope='unit_1')
                net = self.identity_block(inputs=net, filters=[64, 64, 256], stride=1, scope='unit_2')
                net = self.identity_block(inputs=net, filters=[64, 64, 256], stride=1, scope='unit_3')
            # block 2
            # 28 x 28 x 3
            with tf.variable_scope('block2', 'block') as sc:
                net = self.conv_block(inputs=net, filters=[128, 128, 512], stride=2, scope='unit_1')
                net = self.identity_block(inputs=net, filters=[128, 128, 512], stride=1, scope='unit_2')
                net = self.identity_block(inputs=net, filters=[128, 128, 512], stride=1, scope='unit_3')
                net = self.identity_block(inputs=net, filters=[128, 128, 512], stride=1, scope='unit_4')
            # block 3
            # 14 x 14 x 3
            with tf.variable_scope('block3', 'block') as sc:
                net = self.conv_block(inputs=net, filters=[256, 256, 1024], stride=2, scope='unit_1')
                net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_2')
                net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_3')
                net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_4')
                net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_5')
                net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_6')
            # block 4
            # 7 x 7 x 3
            with tf.variable_scope('block4', 'block') as sc:
                net = self.conv_block(inputs=net, filters=[512, 512, 2048], stride=1, scope='unit_1')
                net = self.identity_block(inputs=net, filters=[512, 512, 2048], stride=1, scope='unit_2')
                net = self.identity_block(inputs=net, filters=[512, 512, 2048], stride=1, scope='unit_3')

            return net

    def conv_block(self, inputs, filters, stride=2, rate=1, scope=None):
        """
        conv_block is the block that has a conv layer at shortcut
        :param inputs:
        :param filters:
        :param strides:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope, values=[inputs]) :
            with tf.variable_scope('bottleneck_v2', default_name='bottleneck_v2', values=[inputs]):
                    # get filters num
                    filter0, filter1, filter2 = filters
                    preact = slim.batch_norm(inputs=inputs, activation_fn=tf.nn.relu, scope='preact')
                    # shortcut net
                    shortcut = slim.conv2d(inputs=preact, num_outputs=filter2, kernel_size=[1, 1], stride=stride,
                                           normalizer_fn=None, activation_fn=None, padding='SAME',  scope='shortcut')
                    # stack net
                    net = slim.conv2d(inputs=preact, num_outputs=filter0, kernel_size=[1, 1], stride=1, padding='SAME',
                                      scope='conv1')
                    net = self.conv2d_same(inputs=net, num_outputs=filter1, kernel_size=[3, 3], stride=stride, rate=rate,
                                           scope='conv2')
                    net = slim.conv2d(inputs=net, num_outputs=filter2, kernel_size=[1, 1], stride=1, normalizer_fn=None,
                                      activation_fn=None, padding='SAME', scope='conv3')

                    output = shortcut + net
                    return output

    def identity_block(self, inputs, filters, stride=1, rate=1, scope=None):
        """
        conv_block is the block that has no conv layer at shortcut
        :param inputs:
        :param filters:
        :param stride:
        :param rate:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope, default_name='unit', values=[inputs]):
            with tf.variable_scope('bottleneck_v2', default_name='bottleneck_v2', values=[inputs]):
                with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                    # get filters num
                    filter0, filter1, filter2 = filters
                    preact = slim.batch_norm(inputs=inputs, activation_fn=tf.nn.relu, scope='preact')
                    # shortcut net
                    shortcut = self.subsample(inputs=inputs, factor=stride, scope='shortcut')

                    # stack net
                    net = slim.conv2d(inputs=preact, num_outputs=filter0, kernel_size=[1, 1], stride=1, padding='SAME',
                                      scope='conv1')
                    net = self.conv2d_same(inputs=net, num_outputs=filter1, kernel_size=[3, 3], stride=stride, rate=rate,
                                           scope='conv2')
                    net = slim.conv2d(inputs=net, num_outputs=filter2, kernel_size=[1, 1], stride=1, normalizer_fn=None,
                                      activation_fn=None, padding='SAME', scope='conv3')
                    output = shortcut + net
                    return output


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
            kernel_size_effect = kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)
            pad_total = kernel_size_effect - 1
            pad_begin = pad_total // 2
            pad_end = pad_total - pad_begin
            inputs = tf.pad(inputs, paddings=[[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])

            # int((n - k) / 2) + 1
            return slim.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size,
                               stride=stride, padding='VALID', scope=scope)

    def subsample(self, inputs, factor, scope=None):
        """
        Subsample the input along the spatial dimensions
        :param input:
        :param factor:
        :param scope:
        :return:
        """
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs=inputs, kernel_size=[1, 1], stride=factor, scope=scope)


    def training(self, learnRate, globalStep, loss):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        # define trainable variable
        # trainable_scope = self.trainable_scope
        # trainable_scope = ['resnet_v2_50/Logits']
        trainable_scope = []
        if self.is_pretrain:
            trainable_variable = []
            if trainable_scope is not None:
                for scope in trainable_scope:
                    variables = tf.model_variables(scope=scope)
                    [trainable_variable.append(var) for var in variables]
            else:
                trainable_variable = None
        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op =  tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=globalStep)
        return train_op

    def losses(self, logits, labels, scope='Loss'):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(scope) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='Entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='Entropy_Mean')

    def evaluate_batch(self, logits, labels, scope='Evaluate_Batch'):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        with tf.name_scope(scope):
            correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.float32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict









