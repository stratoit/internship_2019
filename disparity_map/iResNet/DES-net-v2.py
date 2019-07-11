#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
import matplotlib.image as mpimg
import pandas as pd
from keras import backend as K
from keras.utils import plot_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def Corr(x, max_disp=40):
    x1 = x[0]
    x2 = x[1]
    w = tf.shape(x2)[2]
    corr_tensors = []
    for i in range(-max_disp, 0, 1):
        shifted = tf.pad(x2[:, :, 0:w+i, :], [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, x1), axis=3)
        corr_tensors.append(corr)
    for i in range(max_disp + 1):
        shifted = tf.pad(x1[:, :, i:, :], [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, x2), axis=3)
        corr_tensors.append(corr)
    return tf.transpose(tf.stack(corr_tensors), perm=[1, 2, 3, 0])


# ## Model

def desnet_ver2(shape=(None,None,3)):
    
    activation_relu = 'relu'
    init_random_normal = 'random_normal'
    
#     Input Layer
    
    input_left = Input(shape)
    input_right = Input(shape)
    
#     Stem Block for multiscale shared feature extraction begins
    
    conv1 = Conv2D(64, 7, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv1')
    conv1_left = conv1(input_left)
    conv1_right = conv1(input_right)
    
    up1 = Conv2DTranspose(32, 4, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='up1')
    up1_left = up1(conv1_left)
    up1_right = up1(conv1_right)
    
    conv2 = Conv2D(128, 5, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv2')
    conv2_left = conv2(conv1_left)
    conv2_right = conv2(conv1_right)
    
    up2 = Conv2DTranspose(32, 8, strides=4, activation=activation_relu, kernel_initializer=init_random_normal, name='up2')
    up2_left = up2(conv2_left)
    up2_right = up2(conv2_right)
    
    match_conv1 = Conv2D(32, 5, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='matchconv1')
    match_conv1_left = match_conv1(up1_left)
    match_conv1_right = match_conv1(up1_right)
    
    up1_2_left = concatenate([match_conv1_left,up2_left],name='concat_left')
    up1_2_right = concatenate([match_conv1_right,up2_right],name='conact_right')
    
    conv1_2 = Conv2D(32, 1, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='convup1_2')
    conv1_2_left = conv1_2(up1_2_left)
    conv1_2_right = conv1_2(up1_2_right)
    
#     Stem Block for multiscale shared feature extraction ends
    
#     model_shared_features = Model(inputs=[input_left,input_right],outputs=[conv1_2_left,conv1_2_right])
#     model_shared_features.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     model_shared_features.summary()
#     plot_model(model=model_shared_features,show_layer_names=True, show_shapes=True, to_file='model_shared_features_v2.png')
    
#     Initial Disparity Estimation Sub-network begins

    corr1d = Lambda(Corr,arguments={'max_disp':40}, name='corr1d')([conv2_left,conv2_right])
    conv_redir = Conv2D(64, 1, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv_redir')(conv2_left)
    
    corr1d_conv_redir = concatenate([corr1d,conv_redir],name='concat_corr1d_convredir')
    conv3 = Conv2D(256, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv3')(corr1d_conv_redir)
    conv3_1 = Conv2D(256, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv3_1')(conv3)
    
    conv4 = Conv2D(512, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv4')(conv3_1)
    conv4_1= Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv4_1')(conv4)
    
    conv5 = Conv2D(512, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv5')(conv4_1)
    conv5_1 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv5_1')(conv5)
    
    conv6 = Conv2D(1024, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv6')(conv5_1)
    conv6_1= Conv2D(1024, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv6_1')(conv6)
    disp6 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='disp6')(conv6_1)
    
    matchuconv5_disp6 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv5_disp6')(disp6)
    unconv5 = Conv2DTranspose(512,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv5')(conv6_1)
    iconv5_concat1 = concatenate([unconv5,matchuconv5_disp6], name='iconv5_concat1')
    matchuconv5_concat1 = Conv2DTranspose(513,4,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv5_concat1')(iconv5_concat1)
    
    iconv5_concat = concatenate([conv5_1,matchuconv5_concat1], name='iconv5_concat')
    iconv5 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv5')(iconv5_concat)
    disp5 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='disp5')(iconv5)
    
    match_disp5 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_disp5')(disp5)
    unconv4 = Conv2DTranspose(256,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv4')(iconv5)
    iconv4_concat1 = concatenate([unconv4,match_disp5], name='iconv4_concat1')
    match4_concat1 = Conv2DTranspose(257,4,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_1_concat1')(iconv4_concat1)
    match4_concat1 = Conv2DTranspose(257,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_2_concat1')(match4_concat1)
    match4_concat1 = Conv2DTranspose(257,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_3_concat1')(match4_concat1)
    match4_concat1 = ZeroPadding2D(padding=((0,0),(0,1)),name="matchpad4_concat1")(match4_concat1) 
    
    iconv4_concat = concatenate([match4_concat1,conv4_1], name='iconv4_concat')
    iconv4 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv4')(iconv4_concat)
    disp4 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp4')(iconv4)
    
    match_disp4 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv3_disp4')(disp4)
    unconv3 = Conv2DTranspose(128,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv3')(iconv4)
    iconv3_concat1 = concatenate([unconv3,match_disp4], name='iconv3_concat1')
    match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_1_concat1')(iconv3_concat1)
    match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_2_concat1')(match3_concat1)
    match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_3_concat1')(match3_concat1)
    match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_4_concat1')(match3_concat1)
    
    iconv3_concat = concatenate([match3_concat1,conv3_1], name='iconv3_concat')
    iconv3 = Conv2D(128, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv3')(iconv3_concat)
    disp3 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp3')(iconv3)
    
    match_disp3 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv2_disp3')(disp3)
    unconv2 = Conv2DTranspose(64,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv2')(iconv3)
    iconv2_concat1 = concatenate([unconv2,match_disp3], name='iconv2_concat1')
    match2_concat1 = Conv2DTranspose(65,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv2_1_concat1')(iconv2_concat1)
    match2_concat1 = Conv2DTranspose(65,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv2_2_concat1')(match2_concat1)
    match2_concat1 = Conv2DTranspose(65,4,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv2_3_concat1')(match2_concat1)
    
    iconv2_concat = concatenate([conv2_left,match2_concat1], name='iconv2_concat')
    iconv2 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv2')(iconv2_concat)
    disp2 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp2')(iconv2)
    
    match_disp2 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv1_disp2')(disp2)
    unconv1 = Conv2DTranspose(32,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv1')(iconv2)
    iconv1_concat1 = concatenate([unconv1,match_disp2], name='iconv1_concat1')
    match1_concat1 = Conv2DTranspose(33,2,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_1_concat1')(iconv1_concat1)
    match1_concat1 = Conv2DTranspose(33,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_2_concat1')(match1_concat1)
    match1_concat1 = Conv2DTranspose(33,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_3_concat1')(match1_concat1)
    
    iconv1_concat = concatenate([conv1_left,match1_concat1], name='iconv1_concat')
    iconv1 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv1')(iconv1_concat)
    disp1 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp1')(iconv1)
    
    match_disp1 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv0_disp1')(disp1)
    unconv0 = Conv2DTranspose(32,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv0')(iconv1)
    
    iconv0_concat = concatenate([unconv0,match_disp1,conv1_2_left], name='iconv0_concat')
    iconv0 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv0')(iconv0_concat)
    disp0 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp0')(iconv0)
    
#     Initial Disparity Estimation Sub-network ends(DES-Net)
    
    model_init_disp = Model(inputs=[input_left,input_right],outputs=disp0)
    model_init_disp.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model_init_disp.summary()
    plot_model(model=model_init_disp,show_layer_names=True, show_shapes=True, to_file='model_init_disp_v2.png')
    
    return model_init_disp

x = desnet_ver2((1280,720,3))
