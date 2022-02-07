import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa #Group Normalization

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,\
                         UpSampling2D, ZeroPadding2D, InputLayer
from keras import regularizers
from keras import optimizers
from keras import callbacks
from tensorflow.keras import Model
from tensorflow.keras import regularizers

import math

# Ref. https://github.com/pytorch/pytorch/blob/08891b0a4e08e2c642deac2042a02238a4d34c67/torch/nn/modules/conv.py
# based on reset_parameters()
def pytorch_conv2D_initializer(in_channels, kernel_shape, seed):
  n = in_channels
  for k in kernel_shape:
      n *= k
  stdv = 1. / math.sqrt(n)
  return tf.keras.initializers.RandomUniform(minval=-stdv, maxval=stdv, seed=seed)

# RRU-Net Components //////////////////////////////////////////////////////////////////////////////////
## RRU_double_conv -----------------------------------------------------------
class RRU_double_conv(tf.keras.Model):
  def __init__(self, 
                input_channel, output_channel, 
                name='RRU_double_conv', 
                l2_penalty=0, 
                normalization='gn',
                _kernel_initializer="glorot_uniform", _bias_initializer="zeros",
                seed=23):
    super(RRU_double_conv, self).__init__(name=name)

    self.hidden_layers = []

    self.hidden_layers.append(tf.keras.layers.ZeroPadding2D(input_shape=(-1, -1, -1, input_channel), 
																													  padding=(2,2), 
																													  name='zp_1')
                                                          )
    
    if _kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (3, 3), seed)
    else:
      k_init = _kernel_initializer

    if _bias_initializer == 'pytorch':
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (3, 3), seed)
    else:
      b_init = _bias_initializer
    self.hidden_layers.append(tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																									filters=output_channel, 
																									kernel_size=(3,3), 
																									name='c1', 
																									padding='valid', 
																									dilation_rate=(2,2),
                                    							kernel_regularizer=regularizers.l2(l2_penalty),
                                    							bias_regularizer=regularizers.l2(l2_penalty),
                                                  kernel_initializer=k_init,
                                                  bias_initializer=b_init
                                                  ))


    if normalization == 'gn':
      self.hidden_layers.append(tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_1'))
    elif normalization == 'bn':
      self.hidden_layers.append(tf.keras.layers.BatchNormalization(name='bn_1'))
    else:
      print("WARNING!!! No normalization tecnique")

    self.hidden_layers.append(tf.keras.layers.ReLU(name='relu'))

    self.hidden_layers.append(tf.keras.layers.ZeroPadding2D(input_shape=(-1, -1, -1, input_channel), 
																													padding=(2,2), 
																													name='zp_2')
                                                          )

    if _kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (3, 3), seed)
    else:
      k_init = _kernel_initializer

    if _bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (3, 3), seed)
    else:
      b_init = _bias_initializer
    self.hidden_layers.append(tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																									filters=output_channel, 
																									kernel_size=(3,3), 
																									name='c2', 
																									padding='valid', 
																									dilation_rate=(2,2),
                                    							kernel_regularizer=regularizers.l2(l2_penalty),
                                    							bias_regularizer=regularizers.l2(l2_penalty),
                                                  kernel_initializer=k_init,
                                                  bias_initializer=b_init                                                  
                                                  ))
    if normalization == 'gn':
      self.hidden_layers.append(tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_1'))
    elif normalization == 'bn':
      self.hidden_layers.append(tf.keras.layers.BatchNormalization(name='bn_1'))
    else:
      print("WARNING!!! No normalization tecnique")

  @tf.function
  def call(self, x):
    for layer in self.hidden_layers:
      x = layer(x)
    return x
## --------------------------------------------------------------------------
## RRU_first_down -----------------------------------------------------------
class RRU_first_down(tf.keras.Model):
  def __init__(self, 
                input_channel, output_channel, 
                name='RRU_first_down', 
                l2_penalty=0, 
                normalization='gn', 
                kernel_initializer="glorot_uniform", bias_initializer="zeros",
                seed=367):
    super(RRU_first_down, self).__init__(name=name)

    self.conv = RRU_double_conv(input_channel, output_channel, 
																name='dc', 
																l2_penalty=l2_penalty,
																normalization=normalization,
                                _kernel_initializer=kernel_initializer,
                                _bias_initializer=bias_initializer,
                                seed=seed
                                )
    self.relu = tf.keras.layers.ReLU(name='relu')

    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																				filters=output_channel, 
																				kernel_size=(1,1), 
																				name='c1', 
																				padding='valid',
																				use_bias=False,
                                        kernel_regularizer=regularizers.l2(l2_penalty),
                                        bias_regularizer=regularizers.l2(l2_penalty),
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init
                                        )                           

    if normalization == 'gn':
      self.res_conv_gn = tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_1')
    elif normalization == 'bn':
      self.res_conv_gn = tf.keras.layers.BatchNormalization(name='bn_1')
    else:
      print("WARNING!!! No normalization tecnique")
      self.res_conv_gn = None

    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv_back = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, output_channel), 
																						filters=input_channel, 
																						kernel_size=(1,1), 
																						name='c2', 
																						padding='valid', 
																						use_bias=False,
                                            kernel_regularizer=regularizers.l2(l2_penalty),
                                            bias_regularizer=regularizers.l2(l2_penalty),
                                            kernel_initializer=k_init,
                                            bias_initializer=b_init                                   
                                            )
                             
  @tf.function
  def call(self, x):
    # the first ring conv
    ft1 = self.conv(x)
    r1 = self.relu(ft1 + self.res_conv_gn(self.res_conv(x)))

    # the second ring conv
    ft2 = self.res_conv_back(r1)
    x = tf.math.multiply(1+tf.math.sigmoid(ft2), x)

    # the third ring conv
    ft3 = self.conv(x)
    r3 = self.relu(ft3 + self.res_conv_gn(self.res_conv(x)))
    return r3
## ---------------------------------------------------------------------------
## RRU_down -----------------------------------------------------------
class RRU_down(tf.keras.Model):
  def __init__(self, 
                input_channel, output_channel, 
                name='RRU_down', 
                l2_penalty=0, 
                normalization='gn', 
                kernel_initializer="glorot_uniform", bias_initializer="zeros",
                seed=19):
    super(RRU_down, self).__init__(name=name)

    self.padding = tf.keras.layers.ZeroPadding2D(input_shape=(-1, -1, -1, input_channel), 
																							padding=(1,1), 
																							name='zp')
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3), 
																					strides=(2,2), 
																					padding='valid')


    self.conv = RRU_double_conv(input_channel, output_channel, 
																name='dc',
																l2_penalty=l2_penalty,
																normalization=normalization,
                                _kernel_initializer=kernel_initializer,
                                _bias_initializer=bias_initializer,
                                seed=seed    
                                )
    self.relu = tf.keras.layers.ReLU(name='relu')


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																				filters=output_channel, 
																				kernel_size=(1,1), 
																				name='c1', 
																				padding='valid',
																				use_bias=False,
                                        kernel_regularizer=regularizers.l2(l2_penalty),
                                        bias_regularizer=regularizers.l2(l2_penalty),
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init                                            
                                        )                           

    if normalization == 'gn':
      self.res_conv_gn = tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_1')
    elif normalization == 'bn':
      self.res_conv_gn = tf.keras.layers.BatchNormalization(name='bn_1')
    else:
      print("WARNING!!! No normalization tecnique")
      self.res_conv_gn = None


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv_back = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, output_channel), 
																							filters=input_channel, 
																							kernel_size=(1,1), 
																							name='c2', 
																							padding='valid',
																							use_bias=False,
                                            	kernel_regularizer=regularizers.l2(l2_penalty),
                                            	bias_regularizer=regularizers.l2(l2_penalty),
                                              kernel_initializer=k_init,
                                              bias_initializer=b_init                                                  
                                              )
                             
  @tf.function
  def call(self, x):
    x = self.padding(x)
    x = self.pool(x)

    # the first ring conv
    ft1 = self.conv(x)
    r1 = self.relu(ft1 + self.res_conv_gn(self.res_conv(x)))

    # the second ring conv
    ft2 = self.res_conv_back(r1)
    x = tf.math.multiply(1+tf.math.sigmoid(ft2), x)

    # the third ring conv
    ft3 = self.conv(x)
    r3 = self.relu(ft3 + self.res_conv_gn(self.res_conv(x)))
    return r3
## ---------------------------------------------------------------------
## RRU_up -----------------------------------------------------------
class RRU_up(tf.keras.Model):
  def __init__(self, 
                input_channel, output_channel, 
                name='RRU_up', 
                l2_penalty=0, 
                normalization='gn', 
                kernel_initializer="glorot_uniform", bias_initializer="zeros",
                seed=379):
    super(RRU_up, self).__init__(name=name)


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (2, 2), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (2, 2), seed)
    else:
      b_init = bias_initializer
    self.up = tf.keras.layers.Conv2DTranspose(input_shape=(-1, -1, -1, input_channel//2), 
																						filters=input_channel//2, 
																						kernel_size=(2,2), 
																						strides=(2,2),
                                            kernel_regularizer=regularizers.l2(l2_penalty),
                                            bias_regularizer=regularizers.l2(l2_penalty),
                                            kernel_initializer=k_init,
                                            bias_initializer=b_init   
                                            )
    
    if normalization == 'gn':
      self.up_gn = tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_1')
    elif normalization == 'bn':
      self.up_gn = tf.keras.layers.BatchNormalization(name='bn_1')
    else:
      print("WARNING!!! No normalization tecnique")
      self.up_gn = None

    self.conv = RRU_double_conv(input_channel, output_channel, 
																name='dc', 
																l2_penalty=l2_penalty, 
																normalization=normalization,
                                _kernel_initializer=kernel_initializer,
                                _bias_initializer=bias_initializer,
                                seed=seed   
                                )
    self.relu = tf.keras.layers.ReLU(name='relu')


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																				filters=output_channel, 
																				kernel_size=(1,1), 
																				name='c1', 
																				padding='valid', 
																				use_bias=False,
                                        kernel_regularizer=regularizers.l2(l2_penalty),
                                        bias_regularizer=regularizers.l2(l2_penalty),
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init   
                                        )                            
  
    if normalization == 'gn':
      self.res_conv_gn = tfa.layers.GroupNormalization(groups=32, axis=-1, name='gn_2')
    elif normalization == 'bn':
      self.res_conv_gn = tf.keras.layers.BatchNormalization(name='bn_2')
    else:
      print("WARNING!!! No normalization tecnique")
      self.res_conv_gn = None


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.res_conv_back = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, output_channel), 
																							filters=input_channel, 
																							kernel_size=(1,1), 
																							name='c2', 
																							padding='valid', 
																							use_bias=False,
                                              kernel_regularizer=regularizers.l2(l2_penalty),
                                              bias_regularizer=regularizers.l2(l2_penalty),
                                              kernel_initializer=k_init,
                                              bias_initializer=b_init                                                 
                                              )
                              
  @tf.function
  def call(self, x1, x2):
    x1 = self.up_gn(self.up(x1))

    '''
    note:
      torch:
        (batch, channels, height, width)
      
      tf
        (batch, height, width, channel)
    '''
    #print(x2.shape[1])

    diffX = x2.shape[1] - x1.shape[1] #diff heights
    diffY = x2.shape[2] - x1.shape[2] #diff widths

    # tensorflow does not support negative padding
    if diffX < 0:
      x1 = x1[:, -diffX:, :, :]
    else:
      paddings = tf.constant([[0, 0], [diffX, 0], [0, 0], [0, 0]])
      x1 = tf.pad(x1, paddings, "CONSTANT") 

    if diffY < 0:
      x1 = x1[:, :, -diffY:, :]
    else:
      paddings = tf.constant([[0, 0], [0, 0], [diffY, 0], [0, 0]])
      x1 = tf.pad(x1, paddings, "CONSTANT") 

    x = tf.concat([x2, x1], axis=3)
    x = self.relu(x)

    # the first ring conv
    ft1 = self.conv(x)
    r1 = self.relu(self.res_conv_gn(self.res_conv(x)) + ft1)

    # the second ring conv
    ft2 = self.res_conv_back(r1)
    x = tf.math.multiply(1+tf.math.sigmoid(ft2), x)

    # the third ring conv
    ft3 = self.conv(x)
    r3 = self.relu(ft3 + self.res_conv_gn(self.res_conv(x)))
    return r3
## -----------------------------------------------------------------------
## outconv -----------------------------------------------------------
class outconv(tf.keras.Model):
  def __init__(self, 
                input_channel, output_channel, 
                name='outconv', 
                l2_penalty=0, 
                kernel_initializer="glorot_uniform", bias_initializer="zeros",
                seed=557):
    super(outconv, self).__init__(name=name)


    if kernel_initializer == "pytorch":
      seed += 17
      k_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      k_init = kernel_initializer

    if bias_initializer == "pytorch":
      seed += 31
      b_init = pytorch_conv2D_initializer(input_channel, (1, 1), seed)
    else:
      b_init = bias_initializer
    self.conv = tf.keras.layers.Conv2D(input_shape=(-1, -1, -1, input_channel), 
																		filters=output_channel, 
																		kernel_size=(1,1), 
																		name='conv', 
																		padding='valid',
                                    activation='sigmoid', 
                                    kernel_regularizer=regularizers.l2(l2_penalty),
                                    bias_regularizer=regularizers.l2(l2_penalty),
                                    kernel_initializer=k_init,
                                    bias_initializer=b_init
                                    )                           
                 
  @tf.function
  def call(self, x):
    x = self.conv(x)
    return x
## ////////////////////////////////////////////////////////////////////////////////////////////////

## RRU_net //////////////////////////////////////////////////////////////////////////////////////
class RRU_net(Model):
  def __init__(self, 
                n_channels=3, n_classes=1, 
                l2_penalty=0, 
                normalization='gn',
                kernel_initializer="glorot_uniform", bias_initializer="zeros",
                seed_initializer=883):
    super(RRU_net, self).__init__()
  
    seed = seed_initializer
    self.down = RRU_first_down(n_channels, 32, name='down_0', 
                                l2_penalty=l2_penalty, 
                                normalization=normalization, 
                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                seed=seed+17)
    self.down1 = RRU_down( 	32,  64, name='down_1', 
                            l2_penalty=l2_penalty, 
                            normalization=normalization, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            seed=seed+607)
    self.down2 = RRU_down(	64, 128, name='down_2', 
                            l2_penalty=l2_penalty, 
                            normalization=normalization, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            seed=seed+743)
    self.down3 = RRU_down(128,  256, name='down_3', 
                            l2_penalty=l2_penalty, 
                            normalization=normalization, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            seed=seed+71)
    self.down4 = RRU_down(256,  256, name='down_4', 
                            l2_penalty=l2_penalty, 
                            normalization=normalization, 
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            seed=seed+233)

    self.up1 = RRU_up(512, 128, name='up_1', 
                        l2_penalty=l2_penalty, 
                        normalization=normalization, 
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        seed=seed+947)
    self.up2 = RRU_up(256,  64, name='up_2', 
                        l2_penalty=l2_penalty, 
                        normalization=normalization, 
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        seed=seed+431)
    self.up3 = RRU_up(128,  32, name='up_3', 
                        l2_penalty=l2_penalty, 
                        normalization=normalization, 
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        seed=seed+79)
    self.up4 = RRU_up( 64,  32, name='up_4', 
                        l2_penalty=l2_penalty, 
                        normalization=normalization, 
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        seed=seed+887)

    self.out_sigmoid = outconv(32, n_classes, name="out_conv_sigmoid", 
                        l2_penalty=l2_penalty, 
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        seed=seed+43)

  @tf.function
  def call(self, x):
    #print(x.shape)
    x1 = self.down(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)

    # NOTE:
    # In the original paper the output convolutional layer has not an activation
    # function since a sigmoid function is applied manually during the training.
    # For simplicity in this tensorflow implementation we decide to include 
    # directly in the network.
    # Now the output of the network is always between [0, 1] and represents
    # the probabilities of a pixel to belog to a class.
    # In this case (n_classes=1), for each pixel the network returns only one
    # number belongs to [0, 1]
    x = self.out_sigmoid(x)
    return x
    
## ///////////////////////////////////////////////////////////////////////////////////////
