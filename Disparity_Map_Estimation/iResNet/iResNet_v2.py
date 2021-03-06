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

def Warp(x):
    im = x[0]
    disp = x[1]
    b = tf.shape(im)[0]
    h = tf.shape(im)[1]
    w = tf.shape(im)[2]
    c = tf.shape(im)[3]

    disp = tf.squeeze(disp)

    def _warp(i):
        a, y = tf.meshgrid(tf.range(w), tf.range(h))
        x_f = tf.to_float(a)
        x_f -= disp[i]
        x0_f = tf.floor(x_f)
        x1_f = x0_f + 1

        w0 = x1_f - x_f
        w0 = tf.expand_dims(w0, axis=2)
        w1 = x_f - x0_f
        w1 = tf.expand_dims(w1, axis=2)

        x_0 = tf.zeros(shape=[h, w], dtype=tf.float32)
        x_w = tf.ones(shape=[h, w], dtype=tf.float32) * tf.to_float(w - 1)
        x0_f = tf.where(x0_f < 0, x_0, x0_f)
        x0_f = tf.where(x0_f > tf.to_float(w - 1), x_w, x0_f)
        x1_f = tf.where(x1_f < 0, x_0, x1_f)
        x1_f = tf.where(x1_f > tf.to_float(w - 1), x_w, x1_f)

        x0_f = tf.expand_dims(x0_f, axis=2)
        x1_f = tf.expand_dims(x1_f, axis=2)
        y = tf.expand_dims(y, axis=2)
        indices = tf.concat([y, tf.to_int32(x0_f)], axis=2)
        indices = tf.reshape(indices, [-1, 2])
        iml = tf.gather_nd(im[i,:,:,:], indices)
        indices = tf.concat([y, tf.to_int32(x1_f)], axis=2)
        indices = tf.reshape(indices, [-1, 2])
        imr = tf.gather_nd(im[i,:,:,:], indices)

        res = w0 * tf.reshape(iml, [h, w, c]) + w1 * tf.reshape(imr, [h, w, c])
        return res

    ret = tf.map_fn(_warp, tf.range(b), dtype=tf.float32)
    ret = tf.reshape(ret, [b, h, w, c])
    return ret

# ## Model

def iresnet_ver2(shape=(None,None,3)):
    
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
    
#     model_init_disp = Model(inputs=[input_left,input_right],outputs=disp0)
#     model_init_disp.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     model_init_disp.summary()
#     plot_model(model=model_init_disp,show_layer_names=True, show_shapes=True, to_file='model_init_disp_v2.png')
    
#     Disparity Refinement Sub-network begins(iRes-Net)

    match_disp0 = Conv2DTranspose(1,5,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv1_disp0')(disp0)
    w_up_1_2 = Lambda(Warp, name='w_up_1_2')([conv1_2_right,match_disp0])
    w_up_1_2 = Reshape((shape[0]-8,shape[1]-8, 32),name='shape_correct')(w_up_1_2)
    r_diff_conv0 = Subtract(name='r_diff_conv0')([conv1_2_left,w_up_1_2])
    r_abs_conv0 = Lambda(lambda x: abs(x),name='r_abs_conv0')(r_diff_conv0)
    r_concat_conv0 = concatenate([r_abs_conv0,match_disp0,conv1_2_left],name='r_concat_conv0')
    r_conv0 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv0')(r_concat_conv0)
    r_conv1 = Conv2D(64, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv1')(r_conv0)
    
    c_conv1 = Conv2D(16, 4, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='c_conv1')
    c_conv1_left = c_conv1(conv1_left)
    c_conv1_right = c_conv1(conv1_right)
    
    r_corr = Lambda(Corr,arguments={'max_disp':20}, name='r_corr')([c_conv1_left,c_conv1_right])
    
    r_concat_conv1_1 = concatenate([r_conv1,r_corr],name='r_concat_conv1_1')
    r_conv1_1 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv1_1')(r_concat_conv1_1)
    
    r_conv2 = Conv2D(128, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv2')(r_conv1_1)
    r_conv2_1 = Conv2D(128, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv2_1')(r_conv2)
    r_res2 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_res2')(r_conv2_1)
    
    r_uconv1 = Conv2DTranspose(64,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='r_uconv1')(r_conv2_1)
    r_match_res2 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='r_matchupconv_res2')(r_res2)
    r_concat_iconv1_1 = concatenate([r_match_res2,r_uconv1],name='r_concat_iconv1_1')
    r_match1_iconv1 = Conv2DTranspose(65,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match1_1_iconv1')(r_concat_iconv1_1)
    r_match1_iconv1 = Conv2DTranspose(65,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match1_2_iconv1')(r_match1_iconv1)
    
    r_concat_iconv1 = concatenate([r_match1_iconv1,r_conv1_1],name='r_concat_iconv1')
    r_iconv1 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_iconv1')(r_concat_iconv1)
    r_res1 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_res1')(r_iconv1)
    
    r_uconv0 = Conv2DTranspose(32,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='r_uconv0')(r_iconv1)
    r_match_res1 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='r_matchupconv_res1')(r_res1)
    r_concat_iconv0_1 = concatenate([r_match_res1,r_uconv0],name='r_concat_iconv0_1')
    r_match0_iconv0 = Conv2DTranspose(33,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match0_1_iconv0')(r_concat_iconv0_1)
    r_match0_iconv0 = Conv2DTranspose(33,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match0_2_iconv0')(r_match0_iconv0)
    r_match0_iconv0 = Conv2DTranspose(33,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match0_3_iconv0')(r_match0_iconv0)
    r_match0_iconv0 = Conv2DTranspose(33,3,strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_match0_4_iconv0')(r_match0_iconv0)
    
    r_concat_iconv0 = concatenate([r_match0_iconv0,r_conv0],name='r_concat_iconv0')
    r_iconv0 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_iconv0')(r_concat_iconv0)
    r_res0 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_res0')(r_iconv0)
    
    model_refinement = Model(inputs=[input_left,input_right],outputs=r_res0)
    model_refinement.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model_refinement.summary()
    plot_model(model=model_refinement,show_layer_names=True, show_shapes=True, to_file='model_refinement_v2.png')
    
    return model_refinement

x = iresnet_ver2((1280,720,3))