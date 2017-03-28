# --------------------------------------------------------
# TFFRCNN - Resnet50
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by miraclebiu
# --------------------------------------------------------

import tensorflow as tf
from network import Network
import numpy as np


class Pvanet(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.trainable = trainable

    def get(self, inputs, 
            dropout_keep_prob=0.8,                                                                                                                                  
            num_classes=1000,                                                                                                                                       
            is_training=True,                                                                                                                                       
            restore_logits=True,                                                                                                                                    
            scope=''):

        self.layers = {}
        self.layers['data'] = inputs

        with tf.name_scope(scope, 'PVAnet_v9', [inputs]):

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
            
            print("--------3-4")
            print(self.layers['conv3_4'])

            (self.feed('conv3_4')
             .pva_inception_res_block(name = 'conv4_4', name_prefix = 'conv4_', type='a') # downsample
             .pva_inception_res_block(name='conv5_4', name_prefix='conv5_', type='b'))    # downsample

            print("--------5-4")
            print(self.layers['conv5_4'])

            (self.feed('conv5_4')
             #.conv(3, 3, 384, 2, 2, name='conv6', biased=True, relu=False)
             #.max_pool(3, 3, 2, 2, padding='VALID', name='pool6')
             .fc(4096, name='fc6', relu=False)
             .bn_scale_drop_relu_combo(c_in=4096, name='fc6')
             .fc(4096, name='fc7', relu=False)
             .bn_scale_drop_relu_combo(c_in=4096, name='fc7')
             .fc(num_classes, name='fc8', relu=False))

            return self.layers['fc8'], self.layers
