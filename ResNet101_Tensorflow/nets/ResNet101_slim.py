#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File ResNet101_slim.py
# @ Description :
# @ Author alexchung
# @ Time 15/11/2019 AM 09:17


import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ResNet101():
    """
    Inception v1
    """

    def __init__(self, input_shape, num_classes, batch_size, learning_rate, momentum=0.9, num_samples_per_epoch=0,
                 num_epoch_per_decay=0, decay_rate=0.9, weight_decay=0.0001,
                 batch_norm_decay=0.997,batch_norm_epsilon=1e-5, batch_norm_scale=True, batch_norm_fused=True,
                reuse=tf.AUTO_REUSE):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch * num_epoch_per_decay / batch_size)
        self.decay_rate = decay_rate
        self.reuse = reuse
        self.weight_decay = weight_decay
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_norm_scale = batch_norm_scale
        self.batch_norm_fused = batch_norm_fused
        # self._R_MEAN = 123.68
        # self._G_MEAN = 116.78
        # self._B_MEAN = 103.94
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.compat.v1.placeholder(tf.float32,shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                                       name="input_images")
        # self.raw_input_data = self.mean_subtraction(image=self.raw_input_data,
        #                                             means=[self._R_MEAN, self._G_MEAN, self._B_MEAN])
        # y [None,num_classes]
        self.raw_input_label = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")

        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        # self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_step = tf.train.get_or_create_global_step()

        # logits
        self.logits = self.inference(self.raw_input_data, scope='resnet_v2_101')
        # # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, scope='Loss')
        # train operation
        self.train = self.training(learning_rate, momentum, loss=self.loss, global_step=self.global_step)
        self.accuracy = self.get_accuracy(self.logits, self.raw_input_label)

    def inference(self, inputs, scope='resnet_v2_101'):
        """
        Inception V4 net structure
        :param inputs:
        :param scope:
        :return:
        """
        self.prameter = []
        prop = self.resnet101(inputs=inputs,
                             scope=scope,
                             num_classes=self.num_classes,
                             reuse=self.reuse,
                             is_training=self.is_training)

        return prop

    def resnet101(self, inputs, scope='resnet_v2_101F', num_classes=10, reuse=None, is_training=False):
        """
        resnet 101
        :return:
        """
        with slim.arg_scope(resnet_arg_scope(weight_decay=self.weight_decay,
                                             batch_norm_decay=self.batch_norm_decay,
                                             batch_norm_epsilon=self.batch_norm_epsilon,
                                             batch_norm_scale=self.batch_norm_scale)):
            with tf.variable_scope(scope, 'resnet_v2_101', [inputs], reuse=reuse) as sc:

                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = self.resnet101_base(inputs=inputs, scope=sc)
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
                    prop = slim.softmax(logits, scope='Predict')
                    return prop

    def resnet101_base(self, inputs, scope='resnet_v2_101'):
        """
        inception V4 base
        :param inputs:
        :param scope:
        :return:
        """

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
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_7')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_8')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_9')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_10')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_11')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_12')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_13')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_14')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_15')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_16')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_17')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_18')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_19')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_20')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_21')
            net = self.identity_block(inputs=net, filters=[256, 256, 1024], stride=1, scope='unit_22')
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

    def training(self, learing_rate, momentum, loss, global_step, trainable_scope=None):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        # define trainable variable
        # trainable_scope = self.trainable_scope
        # trainable_scope = ['resnet_v2_101/Logits']


        if trainable_scope is not None:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]
        else:
            trainable_variable = None

        # learning rate decay
        learing_rate = tf.train.exponential_decay(learning_rate=learing_rate, global_step=global_step,
                                                  decay_steps=self.decay_steps,
                                                  decay_rate=self.decay_rate)
        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op =  tf.train.MomentumOptimizer(learing_rate, momentum=momentum).minimize(loss,
                                                                                             var_list=trainable_variable,
                                                                                             global_step=global_step)
        return train_op

    def losses(self, logits, labels, scope='Loss'):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(scope) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy')
            loss = tf.reduce_mean(input_tensor=cross_entropy, name='loss')
            weight_loss = tf.add_n(slim.losses.get_regularization_losses())
            total_loss = loss + weight_loss
            tf.summary.scalar("total loss", total_loss)
            return total_loss

    def get_accuracy(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """

        correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

    def load_weights(self, sess, model_path, custom_scope=None):
        """
        load pre train model
        :param sess:
        :param model_path:
        :param custom_scope:
        :return:
        """

        model_variable = tf.model_variables()
        if custom_scope is None:
            custom_scope = ['resnet_v2_101/logits']
        for scope in custom_scope:
            variables = tf.model_variables(scope=scope)
            [model_variable.remove(var) for var in variables]
        saver = tf.train.Saver(var_list=model_variable)
        saver.restore(sess, save_path=model_path)
        print('Successful load pretrain model from {0}'.format(model_path))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict

    def mean_subtraction(self, image, means):
        """
        subtract the means form each image channel (white image)
        :param image:
        :param mean:
        :return:
        """
        num_channels = image.get_shape()[-1]
        image = tf.cast(image, dtype=tf.float32)
        channels = tf.split(value=image, num_or_size_splits=num_channels, axis=3)
        for n in range(num_channels):
            channels[n] -= means[n]
        return tf.concat(values=channels, axis=3, name='concat_channel')


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      activation_fn: The activation function which is used in ResNet.
      use_batch_norm: Whether or not to use batch normalization.
      batch_norm_updates_collections: Collection for the update ops for
        batch norm.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc









