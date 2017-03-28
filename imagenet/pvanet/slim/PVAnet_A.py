from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pvanet.slim import ops
from pvanet.slim import scopes

def pvanet_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
    end_points = {}
    with tf.name_scope(scope, 'pvanet_9', [inputs]):
        with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
            with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                 stride=1, padding='VALID'):
                # Input: 192 x 192 x 3
                end_points['conv1_1'] = pva_neg_blk(inputs, 16, [7, 7], stride=2, scope='conv1_1')
                # 96 x 96 x 32
                net = ops.max_pool(end_points['conv1_1'], [3, 3], stride=2, scope='pool1')

                # 48 x 48 x 32
                end_points['pool1'] = net
                

                ###--------------------------------------Conv2-------------------------------------###
                net = ops.conv2d(net, 24, [1, 1], 1, scope='conv2_1/1')
                net = pva_neg_blk(net, 24, [3, 3], scope='conv2_1/2')
                net = ops.conv2d(net, 64, [1, 1], 1, activation=None, scope='conv2_1/3')
                proj = ops.conv2d(end_points['pool1', 64, [1, 1], 1, relu=False, scope='conv2_1/proj')
                net = tf.add(net, proj, name='conv2_1')
                # 48 x 48 x 64
                end_points['conv2_1'] = net

                net = ops.conv2d(net, 24, [1, 1], 1, scope='conv2_2/1')
                net = pva_neg_blk(net, 24, [3, 3], scope='conv2_2/2')
                net = ops.conv2d(net, 64, [1, 1], 1, activation=None, scope='conv2_2/3')
                net = tf.add(net, end_points['conv2_1'], name='conv2_2')
                # 48 x 48 x 64
                end_points['conv2_2'] = net

                net = ops.conv2d(net, 24, [1, 1], 1, scope='conv2_3/1')
                net = pva_neg_blk(net, 24, [3, 3], scope='conv2_3/2')
                net = ops.conv2d(net, 64, [1, 1], 1, activation=None, scope='conv2_3/3')
                net = tf.add(net, end_points['conv2_2'], name='conv2_3')
                # 48 x 48 x 64
                end_points['conv2_3'] = net


                ###--------------------------------------Conv3-------------------------------------###
                net = ops.conv2d(net, 48, [1, 1], stride=2, scope='conv3_1/1')
                net = pva_neg_blk(net, 48, [3, 3], scope='conv3_1/2')
                net = ops.conv2d(net, 128, [1, 1], 1, activation=None, scope='conv3_1/3')
                proj = ops.conv2d(end_points['conv2_3', 128, [1, 1], stride=2, relu=False, scope='conv2_1/proj')
                net = tf.add(net, proj, name='conv3_1')
                # 24 x 24 x 128
                end_points['conv3_1'] = net

                net = ops.conv2d(net, 48, [1, 1], 1, scope='conv3_2/1')
                net = pva_neg_blk(net, 48, [3, 3], scope='conv3_2/2')
                net = ops.conv2d(net, 128, [1, 1], 1, activation=None, scope='conv3_2/3')
                net = tf.add(net, end_points['conv3_1'], name='conv3_2')
                # 24 x 24 x 128
                end_points['conv3_2'] = net

                net = ops.conv2d(net, 48, [1, 1], 1, scope='conv3_3/1')
                net = pva_neg_blk(net, 48, [3, 3], scope='conv3_3/2')
                net = ops.conv2d(net, 128, [1, 1], 1, activation=None, scope='conv3_3/3')
                net = tf.add(net, end_points['conv3_2'], name='conv3_3')
                # 24 x 24 x 128
                end_points['conv3_3'] = net

                net = ops.conv2d(net, 48, [1, 1], 1, scope='conv3_4/1')
                net = pva_neg_blk(net, 48, [3, 3], scope='conv3_4/2')
                net = ops.conv2d(net, 128, [1, 1], 1, activation=None, scope='conv3_4/3')
                net = tf.add(net, end_points['conv3_3'], name='conv3_4')
                # 24 x 24 x 128
                end_points['conv3_4'] = net


                ###--------------------------------------Conv4-------------------------------------###
                with tf.variable_scope('conv4_1'):
                    proj = ops.conv2d(net, 256, [1, 1], 2, activation=None, scope='proj')
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 2, scope='0')
                        branch3x3 = ops.conv2d(net, 48, [1, 1], 2, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 128, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 24, [1, 1], 2, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_1')
                        pool = ops.max_pool(net, [3, 3], stride=2, scope='pool')
                        pool = ops.conv2d(pool, 128, [1, 1], 1, scope='poolproj')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5, pool])
                    net = ops.conv2d(net, 256, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, proj)
                    # 12 x 12 x 256
                    end_points['conv4_1'] = net

                with tf.variable_scope('conv4_2'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 64, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 128, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 24, [1, 1], 2, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 256, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv4_1'])
                    # 12 x 12 x 256
                    end_points['conv4_2'] = net

                with tf.variable_scope('conv4_3'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 64, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 128, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 24, [1, 1], 2, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 256, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv4_2'])
                    # 12 x 12 x 256
                    end_points['conv4_3'] = net

                with tf.variable_scope('conv4_4'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 64, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 128, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 24, [1, 1], 2, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 48, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 256, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv4_3'])
                    # 12 x 12 x 256
                    end_points['conv4_4'] = net


                ###--------------------------------------Conv5-------------------------------------###
                with tf.variable_scope('conv5_1'):
                    proj = ops.conv2d(net, 384, [1, 1], 2, activation=None, scope='proj')
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 2, scope='0')
                        branch3x3 = ops.conv2d(net, 96, [1, 1], 2, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 192, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 32, [1, 1], 2, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_1')
                        pool = ops.max_pool(net, [3, 3], stride=2, scope='pool')
                        pool = ops.conv2d(pool, 128, [1, 1], 1, scope='poolproj')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5, pool])
                    net = ops.conv2d(net, 384, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, proj)
                    # 6 x 6 x 384
                    end_points['conv5_1'] = net

                with tf.variable_scope('conv5_2'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 96, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 192, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 32, [1, 1], 1, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 384, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv5_1'])
                    # 6 x 6 x 384
                    end_points['conv5_2'] = net

                with tf.variable_scope('conv5_3'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 96, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 192, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 32, [1, 1], 1, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 384, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv5_2'])
                    # 6 x 6 x 384
                    end_points['conv5_3'] = net

                with tf.variable_scope('conv5_4'):
                    with tf.variable_scope('incep'):
                        branch1x1 = ops.conv2d(net, 64, [1, 1], 1, scope='0')
                        branch3x3 = ops.conv2d(net, 96, [1, 1], 1, scope='1_reduce')
                        branch3x3 = ops.conv2d(branch3x3, 192, [3, 3], 1, scope='1_0')
                        branch5x5 = ops.conv2d(net, 32, [1, 1], 1, scope='2_reduce')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_0')
                        branch5x5 = ops.conv2d(branch5x5, 64, [3, 3], 1, scope='2_1')
                        net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch5x5])
                    net = ops.conv2d(net, 384, [1, 1], 1, activation=None, scope='out')
                    net = tf.add(net, end_points['conv5_3'])
                    # 6 x 6 x 384
                    end_points['conv5_4'] = net


                ###--------------------------------------Pool5-------------------------------------###
                ###--------------------------------------FC6-------------------------------------###
                net = ops.fc(net, 4096, scope='fc6')
                end_points['fc6'] = net
                ###--------------------------------------FC7-------------------------------------###
                net = ops.fc(net, 4096, scope='fc7')
                end_points['fc7'] = net
                ###--------------------------------------FC8-------------------------------------###
                net = ops.fc(net, num_classes, batch_norm_params=None, activation=None, scope='fc8')

    return net, end_points


def pva_neg_blk(input,
                   num_filters_out,
                   kernel_size,
                   stride=1,
                   scale=True,
                   negation=True
                   scope=None):
    """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
    with tf.variable_scope(scope):
        conv = ops.conv2d(input, num_filters_out, kernel_size, stride, scope='conv', batch_norm_params=None, activation=None)
        conv = ops.batch_norm(conv, decay=0.9997, scale=False, activation=None, scope="bn")
        out_num = num_filters_out

        if negation:
            conv_neg = tf.multiply(conv, -1.0, name='neg')
            conv = tf.concat(axis=3, values=[conv, conv_neg], name='concat')
            out_num += num_filters_out
        if scale:
            # y = \alpha * x + \beta
            alpha = tf.get_variable('scale/alpha', shape=[out_num,], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('scale/beta', shape=[out_num, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
            # conv = conv * alpha + beta
            conv = tf.add(tf.multiply(conv, alpha), beta)
        return tf.nn.relu(conv, name='relu')

