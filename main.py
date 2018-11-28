#!/usr/bin/env python

"""
My own residual network
Implemented in TF
"""

import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from resnet import ResNet
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#%% Settings
valid_size = 5000
batch_size = 32
pixel_mean = True


#%% Load dataset
def make_onehot(y):
    num_classes = np.max(y)+1
    return (np.arange(num_classes) == y).astype(np.int32)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = make_onehot(y_train)
y_test = make_onehot(y_test)

# Split into train/val
val_idx = np.sort(np.random.choice(y_train.shape[0], size=valid_size, replace=False))
train_idx = np.where(np.isin(np.arange(0, y_train.shape[0]), val_idx) == False)[0]
x_val = x_train[val_idx, :]
y_val = y_train[val_idx, :]

x_train = x_train[train_idx, :]
y_train = y_train[train_idx, :]


def normalize_pictures(x, ma=255., mi=0.):
    x = x.astype(np.float32)
    return (x-mi)/(ma-mi)


def subtract_pixel_mean(x):
    mean = np.mean(x, 0)
    print(mean.shape)
    return x-mean, mean


x_train = normalize_pictures(x_train)
x_val = normalize_pictures(x_val)
x_test = normalize_pictures(x_test)

if pixel_mean:
    x_train, mean = subtract_pixel_mean(x_train)
    x_val = x_val - mean
    x_test = x_test - mean

IMAGE_SIZE = x_train.shape[1]

#%% Show/Store image
fig = plt.figure()
plt.imshow(x_train[100,...])
plt.imsave('img/train_100.png', x_train[100,...])


#%% Create graph
def pad_and_crop(x):
    x = tf.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)))
    bs = tf.shape(x)[0]
    x = tf.random_crop(x, (bs, 32, 32, 3))
    # tf <1.5 --> Need to apply to each image in batch
    return tf.map_fn(lambda y: tf.image.random_flip_left_right(y), x)

num_classes = y_train.shape[1]

graph = tf.Graph()
with graph.as_default():
    # Placeholders
    is_training_pl = tf.placeholder(tf.bool, name='training')
    x_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='batch_data')
    y_pl = tf.placeholder(tf.int32, shape=[None, num_classes], name='batch_labels')

    # Adaptive Learning Rate
    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = 0.01

    # Decay learning rate after 50000 iterations
    # Min. learning rate --> 0.0001
    #learning_rate = tf.cond(tf.less(global_step, tf.constant(1600)), lambda: tf.constant(0.01),
    #                        lambda: tf.train.exponential_decay(start_learning_rate, global_step, 40000, 0.1, staircase=True))
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 50000, 0.1, staircase=True)
    learning_rate = tf.maximum(0.0001, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Data Augmentation --> 0-Padding and random crop
    # Only during training phase, test phase --> Identity
    # tf.summary.image('Image', x_pl)
    img = tf.cond(is_training_pl, lambda: pad_and_crop(x_pl), lambda: tf.identity(x_pl))
    # tf.summary.image('rand_crop_img', img)

    # Get Model & Loss
    model = ResNet()
    logits = model.get_resnet_cifar(img, num_classes, is_training_pl)
    pred_op = tf.argmax(tf.nn.softmax(logits), 1)

    # Loss & Accuracy
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_pl, logits=logits)) + tf.reduce_sum(reg_losses)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('reg_loss', tf.reduce_sum(reg_losses))
    acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pl, 1), pred_op), tf.float32))
    tf.summary.scalar('accuracy', acc_op)

    summary = tf.summary.merge_all()

    # Ensure that BN running averages are updated during training
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
        train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()

#%% Run graph
num_epochs = 150
best_acc = 0


def get_accuracy(pred, labels):
    return 100 * np.sum(pred == np.argmax(labels, 1)) / pred.shape[0]


# GPU growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Run session
with tf.Session(config=config, graph=graph) as sess:
    # Initialize Variables
    sess.run(tf.global_variables_initializer())

    # Create writer for tensorboard
    train_writer = tf.summary.FileWriter('./log_mom/train', sess.graph)
    val_writer = tf.summary.FileWriter('./log_mom/val', sess.graph)

    # Iterate over epochs
    num_train_samples = x_train.shape[0]
    for epoch in range(num_epochs):
        # Shuffle training data
        train_idx = np.random.permutation(np.arange(num_train_samples))
        x_train = x_train[train_idx, ...]
        y_train = y_train[train_idx, ...]
        num_steps = int(np.ceil(num_train_samples/batch_size))
        epoch_loss = 0
        acc = np.array([])

        # Steps within epoch
        for step in range(num_steps):
            start_idx = step*batch_size
            end_idx = np.minimum(num_train_samples, start_idx+batch_size)
            batch_data = x_train[start_idx:end_idx, :, :, :]

            batch_labels = y_train[start_idx:end_idx, :]
            feed_dict = {x_pl: batch_data,
                         y_pl: batch_labels,
                         is_training_pl: True}

            _, batch_loss, batch_classes, batch_summary = sess.run([train_op, loss, pred_op, summary],
                                                                   feed_dict=feed_dict)
            acc = np.append(acc, get_accuracy(batch_classes, batch_labels))
            train_writer.add_summary(batch_summary, global_step.eval())
            epoch_loss += batch_loss
            if (step+1) % 100 == 0:
                # Validation accuracy
                feed_dict = {x_pl: x_val,
                             y_pl: y_val,
                             is_training_pl: False}
                valid_pred, batch_summary = sess.run([pred_op, summary], feed_dict=feed_dict)
                val_acc = get_accuracy(valid_pred, y_val)
                val_writer.add_summary(batch_summary, global_step.eval())
                if val_acc >= best_acc:
                    print('New best accuracy! %.2f %%' % val_acc)
                    best_acc = val_acc
                    saver.save(sess, 'tmp/model_mom.ckpt')
            if (step+1) % 400 == 0:
                print('Current (mean) training accuracy: %.2f %%' % np.mean(acc))

        # Epoch finished
        print('Finished epoch %d of %d' % (epoch + 1, num_epochs))
        print('Avg. epoch loss: %.5f' % (epoch_loss / num_steps))
        epoch_loss = 0

#%% Test Accuracy
def get_accuracy(pred, labels):
    return 100 * np.sum(pred == np.argmax(labels, 1)) / pred.shape[0]

# GPU growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config,graph=graph) as sess:
    saver.restore(sess, 'tmp/model_mom.ckpt')
    print('Model restored!')
    # Test accuracy
    feed_dict = {x_pl: x_test,
                 y_pl: y_test,
                 is_training_pl: False}
    test_pred = sess.run([pred_op], feed_dict=feed_dict)[0]
    print('Testing accuracy: %.2f %%' % get_accuracy(test_pred, y_test))
