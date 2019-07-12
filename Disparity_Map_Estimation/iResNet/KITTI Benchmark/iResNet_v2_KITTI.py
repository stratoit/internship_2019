#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from keras import backend as K
from keras.utils import plot_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preprocessing

# In[3]:


folder_left = os.path.join(os.getcwd(),os.path.join('training','image_2'))
folder_right = os.path.join(os.getcwd(),os.path.join('training','image_3'))
folder_gt = os.path.join(os.getcwd(),os.path.join('training','disp_noc_0'))
files_left = []
files_right = []
files_gt = []
for file in os.listdir(folder_left):
    if(file[-5]=='0'):
        files_left.append(os.path.join(folder_left,file))
        files_right.append(os.path.join(folder_right,file))
        files_gt.append(os.path.join(folder_gt,file))


# ## Train-Test-Val Split

# In[6]:


# train_left, validate_left, test_left = np.split(images_left, [int(.8*len(images_left)), int(.9*len(images_left))])
# train_right, validate_right, test_right = np.split(images_right, [int(.8*len(images_right)), int(.9*len(images_right))])
# train_gt, validate_gt, test_gt = np.split(images_gt, [int(.8*len(images_right)), int(.9*len(images_right))])


# In[7]:


images_left = []
images_right = []
images_gt = []
for file in files_left[0:2]:
    img = Image.open(file)
    images_left.append(np.array(img.resize((1242,375),Image.ANTIALIAS)))
for file in files_right[0:2]:
    img = Image.open(file)
    images_right.append(np.array(img.resize((1242,375),Image.ANTIALIAS)))
for file in files_gt[0:2]:
    img = Image.open(file)
    images_gt.append(np.array(img.resize((1242,375),Image.ANTIALIAS)))
images_left = np.array(images_left)
images_right = np.array(images_right)
images_gt = np.expand_dims(np.array(images_gt),axis=3)


# In[8]:


print(len(images_left))
for image_left,image_right,image_gt in zip(images_left,images_right,images_gt):
    print(image_left.shape,image_right.shape,image_gt.shape)


# In[11]:


plt.imshow(np.squeeze(264-images_gt[1]),cmap='gray')


# ## Correlation Function

# In[ ]:


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


# ## Warping Function

# In[ ]:


def Warp(x):
    im = x[0]
    disp = x[1]
    b = tf.shape(im)[0]
    h = tf.shape(im)[1]
    w = tf.shape(im)[2]
    c = tf.shape(im)[3]

    disp = tf.squeeze(tf.to_float(disp))

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
        iml = tf.to_float(iml)
        indices = tf.concat([y, tf.to_int32(x1_f)], axis=2)
        indices = tf.reshape(indices, [-1, 2])
        imr = tf.gather_nd(im[i,:,:,:], indices)
        imr = tf.to_float(imr)

        res = w0 * tf.reshape(iml, [h, w, c]) + w1 * tf.reshape(imr, [h, w, c])
        return res

    ret = tf.map_fn(_warp, tf.range(b), dtype=tf.float32)
    ret = tf.reshape(ret, [b, h, w, c])
    return ret


# ## Model

# In[ ]:


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
    
    match_up1 = Conv2D(32, (5,7), strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='match_up1')
    match_up1_left = match_up1(up1_left)
    match_up1_right = match_up1(up1_right)
    
    concat_up1_2_left = concatenate([up2_left,match_up1_left],name='concat_up1_2_left')
    concat_up1_2_right = concatenate([up2_right,match_up1_right],name='concat_up1_2_right')
    
    up1_2 = Conv2D(32, 1, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='up1_2')
    up1_2_left = up1_2(concat_up1_2_left)
    up1_2_right = up1_2(concat_up1_2_right)
    
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
    
#     conv4 = Conv2D(512, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv4')(conv3_1)
#     conv4_1= Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv4_1')(conv4)
    
#     conv5 = Conv2D(512, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv5')(conv4_1)
#     conv5_1 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv5_1')(conv5)
    
#     conv6 = Conv2D(1024, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='conv6')(conv5_1)
#     conv6_1= Conv2D(1024, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='conv6_1')(conv6)
#     disp6 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='disp6')(conv6_1)
    
#     matchuconv5_disp6 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv5_disp6')(disp6)
#     unconv5 = Conv2DTranspose(512,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv5')(conv6_1)
#     iconv5_concat1 = concatenate([unconv5,matchuconv5_disp6], name='iconv5_concat1')
#     matchuconv5_concat1 = Conv2DTranspose(513,4,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv5_concat1')(iconv5_concat1)
    
#     iconv5_concat = concatenate([conv5_1,matchuconv5_concat1], name='iconv5_concat')
#     iconv5 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv5')(iconv5_concat)
#     disp5 = Conv2D(1, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='disp5')(conv5_1)
    
#     match_disp5 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_disp5')(disp5)
#     unconv4 = Conv2DTranspose(256,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv4')(conv5_1)
#     iconv4_concat1 = concatenate([unconv4,match_disp5], name='iconv4_concat1')
#     match4_concat1 = Conv2DTranspose(257,4,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv4_1_concat1')(iconv4_concat1)
    
#     iconv4_concat = concatenate([match4_concat1,conv4_1], name='iconv4_concat')
#     iconv4 = Conv2D(512, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv4')(iconv4_concat)
#     disp4 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp4')(iconv4)
    
#     match_disp4 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv3_disp4')(disp4)
#     unconv3 = Conv2DTranspose(128,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv3')(iconv4)
#     iconv3_concat1 = concatenate([unconv3,match_disp4], name='iconv3_concat1')
#     match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_1_concat1')(iconv3_concat1)
#     match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_2_concat1')(match3_concat1)
#     match3_concat1 = Conv2DTranspose(129,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_3_concat1')(match3_concat1)
#     match3_concat1 = Conv2DTranspose(129,2,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv3_4_concat1')(match3_concat1)
    
#     iconv3_concat = concatenate([match3_concat1,conv3_1], name='iconv3_concat')
#     iconv3 = Conv2D(128, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv3')(iconv3_concat)
    disp3 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp3')(conv3_1)
    
    match_disp3 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv2_disp3')(disp3)
    unconv2 = Conv2DTranspose(64,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv2')(conv3_1)
    iconv2_concat1 = concatenate([unconv2,match_disp3], name='iconv2_concat1')
    match2_concat1 = Conv2DTranspose(65,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv2_1_concat1')(iconv2_concat1)
    match2_concat1 = Conv2DTranspose(65,2,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv2_2_concat1')(match2_concat1)
    
    iconv2_concat = concatenate([conv2_left,match2_concat1], name='iconv2_concat')
    iconv2 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv2')(iconv2_concat)
    disp2 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp2')(iconv2)
    
    match_disp2 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv1_disp2')(disp2)
    unconv1 = Conv2DTranspose(32,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv1')(iconv2)
    iconv1_concat1 = concatenate([unconv1,match_disp2], name='iconv1_concat1')
    match1_concat1 = Conv2DTranspose(33,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_1_concat1')(iconv1_concat1)
    match1_concat1 = Conv2DTranspose(33,3,strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_2_concat1')(match1_concat1)
    match1_concat1 = Conv2DTranspose(33,(2,3),strides=1,activation=activation_relu, kernel_initializer=init_random_normal,name='matchuconv1_3_concat1')(match1_concat1)
    
    iconv1_concat = concatenate([conv1_left,match1_concat1], name='iconv1_concat')
    iconv1 = Conv2D(64, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv1')(iconv1_concat)
    disp1 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp1')(iconv1)
    
    match_disp1 = Conv2DTranspose(1,8,strides=2,activation=activation_relu, kernel_initializer=init_random_normal,name='matchupconv0_disp1')(disp1)
    unconv0 = Conv2DTranspose(32,4,strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name= 'unconv0')(iconv1)
    match0_conv1_2_left = ZeroPadding2D(padding=((0,0),(1,1)),name="match0_conv1_2_left")(up1_2_left)     
    
    iconv0_concat = concatenate([unconv0,match_disp1,match0_conv1_2_left], name='iconv0_concat')
    iconv0 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='iconv0')(iconv0_concat)
    disp0 = Conv2D(1, 3, strides=1, activation=activation_relu,kernel_initializer=init_random_normal, name='disp0')(iconv0)
    
#     disp5_final = Conv2DTranspose(1, 7, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='disp5_final')(disp5)
#     disp4_concat = concatenate([disp5_final, disp4], name='disp4_concat')
#     disp4_final = Conv2DTranspose(1, 11, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='disp4_final')(disp4_concat)
#     disp3_concat = concatenate([disp4_final, disp3], name='disp3_concat')
    disp3_final = Conv2DTranspose(1, 7, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='disp3_final')(disp3)
    disp2_concat = concatenate([disp3_final, disp2], name='disp2_concat')
    disp2_final = Conv2DTranspose(1, (9,10), strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='disp2_final')(disp2_concat)
    disp1_concat = concatenate([disp2_final, disp1], name='disp1_concat')
    disp1_final = Conv2DTranspose(1, 4, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='disp1_final')(disp1_concat)
    disp0_concat = concatenate([disp1_final, disp0], name='disp0_concat')
    disp0_final = Conv2DTranspose(1, (5,3), strides=1, activation=activation_relu, kernel_initializer= init_random_normal, name='disp0_final')(disp0_concat)
    
#     Initial Disparity Estimation Sub-network ends(DES-Net)
    
#     model_init_disp = Model(inputs=[input_left,input_right],outputs=disp0)
#     model_init_disp.compile(optimizer = Adam(lr = 0.00001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     model_init_disp.summary()
#     plot_model(model=model_init_disp,show_layer_names=True, show_shapes=True, to_file='model_init_disp_v2.png')
    
#     Disparity Refinement Sub-network begins(iRes-Net)

    w_up_1_2 = Lambda(Warp, name='w_up_1_2')([up1_2_right,disp0_final])
    w_up_1_2 = Reshape((shape[0]-7,shape[1]-10, 32),name='shape_correct')(w_up_1_2)
    
    r_diff_conv0 = Subtract(name='r_diff_conv0')([up1_2_left,w_up_1_2])
    r_abs_conv0 = Lambda(lambda x: abs(x),name='r_abs_conv0')(r_diff_conv0)
    r_concat_conv0 = concatenate([r_abs_conv0,disp0_final,up1_2_left],name='r_concat_conv0')
    r_conv0 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv0')(r_concat_conv0)
    r_conv1 = Conv2D(64, 3, strides=2, activation=activation_relu, kernel_initializer=init_random_normal, name='r_conv1')(r_conv0)
    
    c_conv1 = Conv2D(16, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='c_conv1')
    c_conv1_left = c_conv1(conv1_left)
    c_conv1_right = c_conv1(conv1_right)
    
    c_conv2 = Conv2D(32, 3, strides=1, activation=activation_relu, kernel_initializer=init_random_normal, name='c_conv2')
    c_conv2_left = c_conv2(c_conv1_left)
    c_conv2_right = c_conv2(c_conv1_right)
    
    r_corr = Lambda(Corr,arguments={'max_disp':20}, name='r_corr')([c_conv2_left,c_conv2_right])
    r_match_corr = ZeroPadding2D(padding=((0,1),(0,0)),name="r_match_corr")(r_corr)
    
    r_concat_conv1_1 = concatenate([r_conv1,r_match_corr],name='r_concat_conv1_1')
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
    
    r_final_res2 = Conv2DTranspose(1, 8, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_res2')(r_res2)
    r_concat_res1 = concatenate([r_final_res2,r_res1],name='r_concat_res1')
    r_final_res1 = Conv2DTranspose(1, 12, strides=2, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_res1')(r_concat_res1)
    r_concat_res0 = concatenate([r_final_res1,r_res0],name='r_concat_res0')
    r_final_res0 = Conv2DTranspose(1, (4,5), strides=1, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_1_res0')(r_concat_res0)
    r_final_res0 = Conv2DTranspose(1, (4,5), strides=1, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_2_res0')(r_final_res0)
    r_final_res0 = Conv2DTranspose(1, (4,5), strides=1, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_3_res0')(r_final_res0)
    r_final_res0 = Conv2DTranspose(1, 5, strides=1, activation=activation_relu, kernel_initializer= init_random_normal, name='r_final_4_res0')(r_final_res0)
    
    model_refinement = Model(inputs=[input_left,input_right],outputs=r_final_res0)
    
    return model_refinement


# In[ ]:


x = iresnet_ver2((375,1242,3))
x.compile(optimizer = Adam(lr = 0.00001), loss = 'mean_squared_error', metrics = ['accuracy'])
x.summary()
# plot_model(model=x,show_layer_names=True, show_shapes=True, to_file='model_refinement_v2.png')


# In[ ]:


history = x.fit([images_left,images_right], images_gt , batch_size=2 ,epochs=10 ,verbose=1)


# In[ ]:




