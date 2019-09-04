from abc import abstractmethod
import tensorflow as tf
import numpy as np
import time
from models.layers import conv2d, lrn, max_pool2d, flatten, fully_connected


class ConvNet(object):

    def __init__(self, input_shape, num_classes, **kwargs):
        self.X = tf.placeholder(shape=[None] + input_shape, dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None] + [num_classes], dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)

        self.net = self._build_model(**kwargs)
        self.logits = self.net['logits']
        self.pred = self.net['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        pass

    def predict(self, sess, dataset, **kwargs):
        input_shape = self.net['input'].get_shape()[1]
        batch_size = kwargs.pop('batch_size', 256)

        assert dataset.num_examples > 0, 'Data size should be over 0.'
        steps = dataset.num_examples // batch_size

        y_pred = []
        start_time = time.time()
        for step in range(steps + 1):
            if step == steps:
                _batch_size = dataset.num_examples - steps * batch_size
            else:
                _batch_size = batch_size
            _batch_xs, _ = dataset.next_batch(_batch_size, input_shape=input_shape, shuffle=False, is_training=False)
            _y_pred = sess.run(self.pred,
                               feed_dict={self.X: _batch_xs, self.is_training: False})
            print('Predicting data: {}/{}'.format(step * batch_size + _batch_size, dataset.num_examples), end='\r')
            y_pred.append(_y_pred)
        print('\nPrediction time(s): {:.3f}'.format(time.time() - start_time))
        y_pred = np.concatenate(y_pred, axis=0)

        return y_pred


class AlexNet(ConvNet):

    def _build_model(self, **kwargs):
        """Build the graph of the AlexNet for image classification.

        NOTE: We initialized the neuron biases in the conv2, conv4, conv5, fc6, fc7, and fc8 with the constant 0.1
        instead of 1.0.
        """

        num_classes = int(self.Y.get_shape()[-1])
        input_mean = kwargs.pop('image_mean', 0.0)
        dropout_rate = kwargs.pop('dropout_rate', 0.5)
        dropout_rate = tf.cond(self.is_training, lambda: dropout_rate, lambda: 0.0)

        net = dict()

        # input layer
        net['input'] = self.X - input_mean

        # conv1-pool1
        with tf.variable_scope('conv1'):
            net['conv1'] = conv2d(net['input'], 96, 11, 4, padding='VALID',
                                  weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                  bias_initializer=tf.constant_initializer(value=0.0))
            net['conv1'] = tf.nn.relu(net['conv1'])
            net['conv1'] = lrn(net['conv1'], depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
        net['pool1'] = max_pool2d(net['conv1'], 3, 2, padding='VALID')

        # conv2-pool2
        with tf.variable_scope('conv2'):
            net['conv2'] = conv2d(net['pool1'], 256, 5, 1, padding='SAME',
                                  weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                  bias_initializer=tf.constant_initializer(value=0.1))
            net['conv2'] = tf.nn.relu(net['conv2'])
            net['conv2'] = lrn(net['conv2'], depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
        net['pool2'] = max_pool2d(net['conv2'], 3, 2, padding='VALID')

        # conv3
        with tf.variable_scope('conv3'):
            net['conv3'] = conv2d(net['pool2'], 384, 3, 1, padding='SAME',
                                  weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                  bias_initializer=tf.constant_initializer(value=0.0))
            net['conv3'] = tf.nn.relu(net['conv3'])

        # conv4
        with tf.variable_scope('conv4'):
            net['conv4'] = conv2d(net['conv3'], 384, 3, 1, padding='SAME',
                                  weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                  bias_initializer=tf.constant_initializer(value=0.1))
        net['conv4'] = tf.nn.relu(net['conv4'])

        # conv5-pool5-flat5
        with tf.variable_scope('conv5'):
            net['conv5'] = conv2d(net['conv4'], 256, 3, 1, padding='SAME',
                                  weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                  bias_initializer=tf.constant_initializer(value=0.1))
        net['conv5'] = tf.nn.relu(net['conv5'])
        net['pool5'] = max_pool2d(net['conv5'], 3, 2, padding='VALID')
        net['flat5'] = flatten(net['pool5'])

        # fc6
        with tf.variable_scope('fc6'):
            net['fc6'] = fully_connected(net['flat5'], 4096,
                                         weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.constant_initializer(value=0.1))
            net['fc6'] = tf.nn.relu(net['fc6'])
            net['fc6'] = tf.nn.dropout(net['fc6'], rate=dropout_rate)

        # fc7
        with tf.variable_scope('fc7'):
            net['fc7'] = fully_connected(net['fc6'], 4096,
                                         weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.constant_initializer(value=0.1))
            net['fc7'] = tf.nn.relu(net['fc7'])
            net['fc7'] = tf.nn.dropout(net['fc7'], rate=dropout_rate)

        # fc8
        with tf.variable_scope('fc8'):
            net['logits'] = fully_connected(net['fc7'], num_outputs=num_classes,
                                            weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                            bias_initializer=tf.constant_initializer(value=0.1))
        # softmax
        net['pred'] = tf.nn.softmax(net['logits'])

        print('layer\tout_shape')
        for i, key in enumerate(net.keys()):
            print('{}\t{}'.format(key, net[key].shape.as_list()[1:]))

        print('\nvariable\tshape')
        var_list = tf.trainable_variables()
        for var in var_list:
            print('{}\t{}'.format(var.name, var.shape.as_list()))

        return net

    def _build_loss(self, **kwargs):
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        var_list = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in var_list])
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,
                                                                                    logits=self.logits))
        return cross_entropy_loss + weight_decay * l2_loss
