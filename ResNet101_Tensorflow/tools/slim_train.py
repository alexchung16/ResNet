#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File ResNet101_train.py
# @ Description :
# @ Author alexchung
# @ Time 21/11/2019 PM 15:44

import os
from datetime import datetime
import numpy as np
import tensorflow as tf

from ResNet101_Tensorflow.nets.ResNet101_slim import ResNet101
from DataProcess.read_TFRecord import dataset_tfrecord, get_num_samples

# dataset path
dataset_dir = '/home/alex/Documents/dataset/flower_tfrecord'
train_data_path = os.path.join(dataset_dir, 'train')
test_data_path = os.path.join(dataset_dir, 'val')

# pretrain model
pretrain_model_path = '/home/alex/Documents/pretrain_model/resnet/resnet_101/resnet_v2_101.ckpt'

# output path
logs_dir = os.path.join('../', 'outputs', 'logs')
model_dir = save_dir = os.path.join('../', 'outputs', 'model')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 299, 'Number of height size.')
flags.DEFINE_integer('width', 299, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 5, 'Number of image class.')
flags.DEFINE_integer('batch_size', 32, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 60, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.9, 'momentum rate.')
flags.DEFINE_float('decay_rate', 0.1, 'Number of learning decay rate.')
flags.DEFINE_integer('num_epoch_per_decay', 20, 'Number epoch after each leaning rate decay.')

flags.DEFINE_float('weight_decay', 0.0001, 'Number of weight decay with regular')
flags.DEFINE_float('batch_norm_decay', 0.997, 'Number of batch norm decay.')
flags.DEFINE_float('batch_norm_epsilon', 1e-5, 'Number of batch norm epsilon.')
flags.DEFINE_bool('batch_norm_scale', True, 'if True, use gamma for update.')
flags.DEFINE_bool('batch_norm_fused', True, 'if True, use a faster, fused implementation if possible.')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')

flags.DEFINE_string('train_data_dir', train_data_path, 'Directory to put the training data.')
flags.DEFINE_string('test_data_dir', test_data_path, 'Directory to put the training data.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
flags.DEFINE_string('model_dir', model_dir, 'direct of summary model to save.')
flags.DEFINE_integer('save_step_period', 2000, 'save model step period')
flags.DEFINE_string('pretrain_model_path', pretrain_model_path, 'pretrain model path.')

def predict(model_name, image_data, input_op_name, predict_op_name):
    """
    model read and predict
    :param model_name:
    :param image_data:
    :param input_op_name:
    :param predict_op_name:
    :return:
    """
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_name, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        image = image_data.eval()
        input = sess.graph.get_tensor_by_name(name=input_op_name)
        output = sess.graph.get_tensor_by_name(name=predict_op_name)

        predict_softmax = sess.run(fetches=output, feed_dict={input: image})
        predict_label = np.argmax(predict_softmax, axis=1)
        return predict_label


# test build network
def networkStructureTest(batch_size):
    inputs = tf.random_uniform(shape=(batch_size, FLAGS.height, FLAGS.width, FLAGS.depth))
    labels = tf.random_uniform(shape=(batch_size, FLAGS.num_classes))

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)

        inputs, labels = sess.run([inputs, labels])
        feed_dict = resnet_101.fill_feed_dict(image_feed=inputs, label_feed=labels, is_training=True)

        logits = sess.run(fetches=[resnet_101.logits], feed_dict=feed_dict)
        assert list(logits[0].shape) == [batch_size, FLAGS.num_classes]

#
#
if __name__ == "__main__":

    num_train_samples = get_num_samples(record_dir=FLAGS.train_data_dir)
    num_val_samples = get_num_samples(record_dir=FLAGS.test_data_dir)

    # get total step of the number train epoch
    step_per_epoch = num_train_samples // FLAGS.batch_size  # get num step of per epoch
    max_step = FLAGS.epoch * step_per_epoch  # get total step of several epoch

    resnet_101 = ResNet101(input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                               num_classes=FLAGS.num_classes,
                               batch_size=FLAGS.batch_size,
                               learning_rate=FLAGS.learning_rate,
                               momentum=FLAGS.momentum,
                               decay_rate=FLAGS.decay_rate,
                               num_samples_per_epoch=num_train_samples,
                               num_epoch_per_decay=FLAGS.num_epoch_per_decay,
                               weight_decay=FLAGS.weight_decay,
                               batch_norm_decay=FLAGS.batch_norm_decay,
                               batch_norm_epsilon=FLAGS.batch_norm_epsilon,
                               batch_norm_scale=FLAGS.batch_norm_scale,
                               batch_norm_fused=FLAGS.batch_norm_fused)

    # add scalar value to summary protocol buffer
    tf.summary.scalar('loss', resnet_101.loss)
    tf.summary.scalar('accuracy', resnet_101.accuracy)

    train_images_batch, train_labels_batch, train_filenames = dataset_tfrecord(record_files=FLAGS.train_data_dir,
                                                                               batch_size=FLAGS.batch_size,
                                                                               target_shape=[FLAGS.height, FLAGS.width,
                                                                                             FLAGS.depth],
                                                                               class_depth=FLAGS.num_classes,
                                                                               epoch=FLAGS.epoch,
                                                                               shuffle=True,
                                                                               is_training=True)
    val_images_batch, val_labels_batch, val_filenames = dataset_tfrecord(record_files=FLAGS.test_data_dir,
                                                                         batch_size=FLAGS.batch_size,
                                                                         target_shape=[FLAGS.height, FLAGS.width,
                                                                                       FLAGS.depth],
                                                                         class_depth=FLAGS.num_classes,
                                                                         epoch=FLAGS.epoch,
                                                                         shuffle=True,
                                                                         is_training=False)
    saver = tf.train.Saver()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu101% of MEM
    config.gpu_options.allow_growth = True
    # train and save model
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # get computer graph
        graph = tf.get_default_graph()
        logit_op = resnet_101.logits.op.name
        write = tf.summary.FileWriter(logdir=FLAGS.logs_dir, graph=graph)
        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.op.name)
            print(var.shape)
        # get and add histogram to summary protocol buffer
        logit_weight = graph.get_tensor_by_name(name='resnet_v2_101/logits/weights:0')
        tf.summary.histogram(name='logits/weight', values=logit_weight)
        logit_biases = graph.get_tensor_by_name(name='resnet_v2_101/logits/biases:0')
        tf.summary.histogram(name='logits/biases', values=logit_biases)
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            resnet_101.load_weights(sess, model_path=FLAGS.pretrain_model_path)

        # print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():

                # ++++++++++++++++++++++++++++++++++start training+++++++++++++++++++++++++++++++++++++++++++++++++
                # used to record the number step of per epoch
                step_epoch = 0
                for step in range(max_step):
                    # --------------------------------print number of epoch--------------------------------------
                    if (step) % step_per_epoch == 0:
                        tmp_epoch = (step + 1) // step_per_epoch
                        print('Epoch: {0}/{1}'.format(tmp_epoch + 1, FLAGS.epoch))

                    # +++++++++++++++++++++++++++++++train part++++++++++++++++++++++++++++++++++++++++++++++++
                    train_image, train_label = sess.run([train_images_batch, train_labels_batch])

                    feed_dict = resnet_101.fill_feed_dict(image_feed=train_image, label_feed=train_label,
                                                         is_training=True)

                    _, train_loss, train_accuracy, summary = sess.run(fetches=[resnet_101.train, resnet_101.loss,
                                                                               resnet_101.accuracy, summary_op],
                                                                      feed_dict=feed_dict)
                    write.add_summary(summary=summary, global_step=step)

                    step_epoch += 1
                    # print training info
                    print(
                        '\tstep {0}:loss value {1}  train accuracy {2}'.format(step_epoch, train_loss, train_accuracy))

                    # -------------------------save_model every  save_step_period--------------------------------
                    if (step + 1) % FLAGS.save_step_period == 0:
                        saver.save(sess, save_path=os.path.join(FLAGS.model_dir, 'model.ckpt'),
                                   global_step=resnet_101.global_step)
                    # ++++++++++++++++++++++++++++++++validation part++++++++++++++++++++++++++++++++++++++++++++
                    # execute validation when complete every epoch
                    # validation use with all validation dataset
                    if (step + 1) % step_per_epoch == 0:
                        val_losses = []
                        val_accuracies = []
                        val_max_steps = int(num_val_samples / FLAGS.batch_size)
                        for _ in range(val_max_steps):
                            val_images, val_labels = sess.run([val_images_batch, val_labels_batch])

                            feed_dict = resnet_101.fill_feed_dict(image_feed=val_images, label_feed=val_labels,
                                                                 is_training=False)

                            val_loss, val_acc = sess.run([resnet_101.loss, resnet_101.accuracy],
                                                         feed_dict=feed_dict)

                            val_losses.append(val_loss)
                            val_accuracies.append(val_acc)
                        mean_loss = np.array(val_losses, dtype=np.float32).mean()
                        mean_acc = np.array(val_accuracies, dtype=np.float32).mean()

                        print("\t{0}: epoch: {1}  val Loss: {2}, val accuracy:  {3}".format(datetime.now(),
                                                                                            (step + 1) // step_per_epoch,
                                                                                            mean_loss, mean_acc))
                        step_epoch = 0  # update step_epoch

                saver.save(sess, save_path=os.path.join(FLAGS.model_dir, 'model.ckpt'),
                           global_step=resnet_101.global_step)
                write.close()


                # ++++++++++++++++++++++++++++save model to pb+++++++++++++++++++++++++++++++++++++++
                # get op name for save model
                # input_op = vgg.raw_input_data.name
                # logit_op = vgg.logits.name
                input_op = resnet_101.raw_input_data.op.name
                logit_op = resnet_101.logits.op.name
                # convert variable to constant
                input_graph_def = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                              [input_op, logit_op])
                # save to serialize file
                with tf.gfile.FastGFile(name=os.path.join(FLAGS.model_dir, 'model.pb'), mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')
