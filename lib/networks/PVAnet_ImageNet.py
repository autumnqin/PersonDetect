# --------------------------------------------------------
# TFFRCNN - Resnet50
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by miraclebiu
# --------------------------------------------------------

import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg
import numpy as np


class PVAnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes, \
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]
        (self.feed('data')
         .pva_negation_block(7, 7, 16, 2, 2, name='conv1_1', negation=True)         # downsample
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')                       # downsample
         .conv(1, 1, 24, 1, 1, name='conv2_1/1/conv', biased=True, relu=False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_1/2', negation=False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_1/3', negation=True))

        (self.feed('pool1')
         .conv(1,1, 64, 1, 1, name='conv2_1/proj', relu=True))

        (self.feed('conv2_1/3', 'conv2_1/proj')
         .add(name='conv2_1')
         .pva_negation_block_v2(1, 1, 24, 1, 1, 64, name='conv2_2/1', negation = False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_2/2', negation = False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_2/3', negation = True))

        (self.feed('conv2_2/3', 'conv2_1')
         .add(name='conv2_2')
         .pva_negation_block_v2(1, 1, 24, 1, 1, 64, name='conv2_3/1', negation=False)
         .pva_negation_block_v2(3, 3, 24, 1, 1, 24, name='conv2_3/2', negation=False)
         .pva_negation_block_v2(1, 1, 64, 1, 1, 24, name='conv2_3/3', negation=True))

        (self.feed('conv2_3/3', 'conv2_2')
         .add(name='conv2_3')
         .pva_negation_block_v2(1, 1, 48, 2, 2, 64, name='conv3_1/1', negation=False) # downsample
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_1/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_1/3', negation=True))

        (self.feed('conv3_1/1/relu')
         .conv(1, 1, 128, 2, 2, name='conv3_1/proj', relu=True))

        (self.feed('conv3_1/3', 'conv3_1/proj')  # 128
         .add(name='conv3_1')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_2/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_2/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_2/3', negation=True))

        (self.feed('conv3_2/3', 'conv3_1')  # 128
         .add(name='conv3_2')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_3/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_3/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_3/3', negation=True))

        (self.feed('conv3_3/3', 'conv3_2')  # 128
         .add(name='conv3_3')
         .pva_negation_block_v2(1, 1, 48, 1, 1, 128, name='conv3_4/1', negation=False)
         .pva_negation_block_v2(3, 3, 48, 1, 1, 48, name='conv3_4/2', negation=False)
         .pva_negation_block_v2(1, 1, 128, 1, 1, 48, name='conv3_4/3', negation=True))

        (self.feed('conv3_4/3', 'conv3_3')  # 128
         .add(name='conv3_4')
         .max_pool(3, 3, 2, 2, padding='SAME', name='downsample')) # downsample

        (self.feed('conv3_4')
         .pva_inception_res_block(name = 'conv4_4', name_prefix = 'conv4_', type='a') # downsample
         .pva_inception_res_block(name='conv5_4', name_prefix='conv5_', type='b'))    # downsample

        (self.feed('conv5-4')
         .fc(4096, name='fc6', relu=False)
         .bn_scale_drop_relu_combo(c_in=4096, name='fc6')
         .fc(4096, name='fc7', relu=False)
         .bn_scale_drop_relu_combo(c_in=4096, name='fc7')
         .fc(1000, name='fc8', relu=False))
