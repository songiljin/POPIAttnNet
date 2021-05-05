from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation,GRU,Bidirectional
from tensorflow.keras.layers import AveragePooling1D, Input, GlobalAveragePooling1D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import config
from model import POPIAttnNet
#K.set_learning_phase(1)


def abs_backend(inputs):
    return K.abs(inputs)
 
def expand_dim_backend(inputs):
    return K.expand_dims(inputs,1)
 
def sign_backend(inputs):
    return K.sign(inputs)
 
 
# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
     
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
     
    for i in range(nb_blocks):
         
        identity = residual
         
        if not downsample:
            downsample_strides = 1
         
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, strides=downsample_strides, 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
         
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
         
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)
         
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
         
        # Calculate thresholds
        thres = tf.keras.layers.multiply([abs_mean, scales])
         
        # Soft thresholding
        sub = tf.keras.layers.subtract([residual_abs, thres])
        zeros = tf.keras.layers.subtract([sub, sub])
        n_sub = tf.keras.layers.maximum([sub, zeros])
        residual = tf.keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
         
        # Downsampling (it is important to use the pooL-size of (1, 1))
        if downsample_strides > 1:
            identity = AveragePooling1D(pool_size=1, strides=2)(identity)
            #identity=Conv1D(out_channels,kernel_size=1,strides=2,padding='same')(identity) 
        # Zero_padding to match channels (it is important to use zero padding rather than 1by1 convolution)
        #if in_channels != out_channels:
         #   identity = Lambda(pad_backend)(identity,in_channels,out_channels)
        else:
            identity=lambda x:x
        #residual = tf.keras.layers.add([residual, identity])
     
    return residual

# define and train a model
def model_1():
    inputs = Input(shape=config.input_shape)
    net = Bidirectional( GRU(
                 512,return_sequences=True,recurrent_initializer='glorot_uniform',dropout=0.5,
                 recurrent_dropout=0.5,kernel_regularizer=tf.keras.regularizers.l2(0.001)))(inputs)
    net = Dropout(0.5)(net)
    net = residual_shrinkage_block(net, 1, 512)
    #net = residual_shrinkage_block(net, 1, 512 )
    #net = residual_shrinkage_block(net, 1, 256) 
    net = residual_shrinkage_block(net, 1, 256 )
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    
    model_popi =  POPIAttnNet(config.w1_units,config.mhcip_units,config.BATCH_SIZE,config.fc_dim)
    hidden = model_popi.initialize_hidden_state()
    outputs,attention_weights = model_popi(net,hidden)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
model=model_1()
model.summary()
#model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))