"""

"""

import tensorflow as tf
import numpy as np


def res_block(x, scope, kernel_size, out_channels, training=False, pool=False, first=False):
    # Get input channels
    in_channels = x.get_shape()[-1]

    # Check, if feature map size changes
    if pool:
        stride = [1, 2, 2, 1]
    else:
        stride = [1, 1, 1, 1]

    # Get variables
    with tf.variable_scope(scope) as sc:
        w1 = tf.get_variable('w1', shape=[kernel_size, kernel_size, in_channels, out_channels],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        b1 = tf.get_variable('b1', shape=[out_channels], dtype=tf.float32, initializer=tf.initializers.zeros())
        w2 = tf.get_variable('w2', shape=[kernel_size, kernel_size, out_channels, out_channels],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        b2 = tf.get_variable('b2', shape=[out_channels], dtype=tf.float32, initializer=tf.initializers.zeros())

    # Define layers
    # Pre-Activation approach BN --> ReLU --> Conv
    if not first:
        net = tf.layers.batch_normalization(x, training=training)
        net = tf.nn.relu(net)
    else:
        net = x
    net = tf.nn.conv2d(net, w1, stride, 'SAME') + b1
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, w2, [1, 1, 1, 1], 'SAME') + b2

    # Add identity
    if (in_channels != out_channels) | pool:
        # Rescale using 1x1-conv, if #channels is different
        with tf.variable_scope(scope) as sc:
            ws = tf.get_variable('ws', shape=[1, 1, in_channels, out_channels],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            return net + tf.nn.conv2d(x, ws, stride, 'SAME')
    else:
        return net + x


def bottleneck_block(x, scope, kernel_size, out_channels, training=False, pool=False, first = False):
    # Get input channels
    in_channels = tf.shape(x)[-1]

    # Check, if feature map size changes
    if pool:
        stride = [1, 2, 2, 1]
    else:
        stride = [1, 1, 1, 1]

    # Get variables
    with tf.variable_scope(scope) as sc:
        w1 = tf.get_variable('w1', shape=[1, 1, in_channels, out_channels//4],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        b1 = tf.get_variable('b1', shape=[out_channels//4], dtype=tf.float32, initializer=tf.initializers.zeros())
        w2 = tf.get_variable('w2', shape=[kernel_size, kernel_size, out_channels//4, out_channels//4],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        b2 = tf.get_variable('b2', shape=[out_channels//4], dtype=tf.float32, initializer=tf.initializers.zeros())
        w3 = tf.get_variable('w3', shape=[1, 1, out_channels//4, out_channels],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        b3 = tf.get_variable('b3', shape=[out_channels], dtype=tf.float32, initializer=tf.initializers.zeros())

    # Define layers
    # Pre-Activation approach BN --> ReLU --> Conv
    if not first:
        net = tf.layers.batch_normalization(x, training=training)
        net = tf.nn.relu(net)
    else:
        net = x

    net = tf.nn.conv2d(net, w1, stride, 'SAME') + b1
    net = tf.layers.batch_normalization(net,training=training)
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, w2, [1, 1, 1, 1], 'SAME') + b2
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, w3, [1, 1, 1, 1], 'SAME') + b3

    # Add identity
    if (in_channels != out_channels) | pool:
        # Rescale using 1x1-conv, if #channels is different
        with tf.variable_scope(scope) as sc:
            ws = tf.get_variable('ws', shape=[1, 1, in_channels, out_channels],
                                 initializer=tf.contrib.layers.xavier_intializer(),
                                 regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            return net + tf.nn.conv2d(x, ws, stride, 'SAME')
    else:
        return net + x


class ResNet():
    # Constructor --> ResNet-110 for CIFAR-10 are default values
    def __init__(self, kernel_size=3, block_size=[18, 18, 18], bottleneck=False, block_channels=[16, 32, 64]):
        self.bottleneck = bottleneck
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.block_channels = block_channels
        self.num_blocks = len(block_size)

    def get_resnet_cifar(self, x, num_classes, is_training):
        # First conv layer
        with tf.variable_scope('conv1'):
            w = tf.get_variable('w', shape=[3, 3, 3, 16], initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            b = tf.get_variable('b', shape=[16], initializer=tf.initializers.zeros())

        # Output: Nonex32x32x16
        net = tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')+b
        net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training))

        # Bottleneck-Blocks
        for i in range(self.num_blocks):
            num_layers = self.block_size[i]*2
            print('Building ResNet block %d with %d layers' % (i+1, num_layers))
            for j in range(self.block_size[i]):
                if (j == 0) & (i != 0):
                    # Pool at first layer of block
                    # Except for first block
                    pool = True
                else:
                    pool = False
                if (j is 0) & (i is 0):
                    first = True
                else:
                    first = False

                net = res_block(net, 'conv'+str(i+2)+'_'+str(j), self.kernel_size, self.block_channels[i],
                                training=is_training, pool=pool, first=first)
                print(net.get_shape())

        # Average Pooling & softmax-block
        net = tf.reduce_mean(net, axis=[1, 2])
        #print(net.get_shape())
        net = tf.squeeze(net)
        with tf.variable_scope('fc'):
            w = tf.get_variable('w', shape=[64, num_classes], initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            b = tf.get_variable('b', shape=[num_classes], initializer=tf.initializers.zeros())
        return tf.matmul(net, w)+b
