import os
from collections import OrderedDict

import tensorflow as tf
slim = tf.contrib.slim

def loss_vgg(images, reuse=False, pooling='max', subtract_mean=True, gram_layers=[], final_endpoint='pool5'):

    filter_size = [3, 3]
    conv1 = lambda net, name: slim.conv2d(net, 64, filter_size, scope=name)
    conv2 = lambda net, name: slim.conv2d(net, 128, filter_size, scope=name)
    conv3 = lambda net, name: slim.conv2d(net, 256, filter_size, scope=name)
    conv4 = lambda net, name: slim.conv2d(net, 512, filter_size, scope=name)
    conv5 = conv4
    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pool =  lambda net, name: pooling_fns[pooling](net, [2, 2], scope=name)
    dropout = lambda net, name: slim.dropout(net, 0.5, is_training=False, scope=name)

    layers = OrderedDict()
    layers['conv1/conv1_1'] = conv1
    layers['conv1/conv1_2'] = conv1
    layers['pool1'] = pool
    layers['conv2/conv2_1'] = conv2
    layers['conv2/conv2_2'] = conv2
    layers['pool2'] = pool
    layers['conv3/conv3_1'] = conv3
    layers['conv3/conv3_2'] = conv3
    layers['conv3/conv3_3'] = conv3
    layers['conv3/conv3_4'] = conv3
    layers['pool3'] = pool
    layers['conv4/conv4_1'] = conv4
    layers['conv4/conv4_2'] = conv4
    layers['conv4/conv4_3'] = conv4
    layers['conv4/conv4_4'] = conv4
    layers['pool4'] = pool
    layers['conv5/conv5_1'] = conv5
    layers['conv5/conv5_2'] = conv5
    layers['conv5/conv5_3'] = conv5
    layers['conv5/conv5_4'] = conv5
    layers['pool5'] = pool
    layers['fc6'] = lambda net, name: slim.conv2d(net, 4096, [7, 7], padding='VALID', scope=name)
    layers['dropout6'] =  dropout
    layers['fc7'] = lambda net, name: slim.conv2d(net, 4096, [1, 1], padding='VALID', scope=name)
    layers['dropout7'] =  dropout
    layers['fc8'] = lambda net, name: slim.conv2d(net, 1000, [1, 1], padding='VALID', scope=name)

    with tf.variable_scope('vgg_19', reuse=reuse) as sc:
        net = images
        if subtract_mean:
            net -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d], trainable=False):
            for layer_name, layer_op in layers.items():
                net = layer_op(net, layer_name)
                end_points[layer_name] = net
                if final_endpoint == layer_name:
                    break
            for layer_name in gram_layers:
                gram = gram_matrix(end_points[layer_name])
                end_points[layer_name + '/gram_matrix'] = gram

    return end_points


def gram_matrix(x):
    return tf.reduce_sum(tf.batch_matmul(x, x, adj_x=True), 1) / tf.cast(tf.reduce_prod(tf.shape(x)[1:4]), tf.float32)
