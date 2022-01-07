import os
import tensorflow as tf
from keras import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten, BatchNormalization, SpatialDropout1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
#from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

from numpy.random import seed
seed(100)
tf.random.set_seed(100)

tf.config.run_functions_eagerly(True)
#os.environ['PROJ_LIB'] = "D:/anaconda3/envs/gpu-enabled-spatial/Library/share/proj"
physical_devices = tf.config.list_physical_devices("GPU")
#print("Num GPUs Available: ", len(physical_devices))

#%% Defined convLSTM block and the inception blocks for complex model
def conv_lstm_module(input_layer,input_length,conv_filters,lstm_filters,kern1,kern2):    
    b1 = Conv1D(conv_filters, kernel_size=kern1, activation='relu',padding='same')(input_layer)
    b1 = Conv1D(conv_filters, kernel_size=kern1, activation='relu',padding='same')(b1)
    b1 = Conv1D(conv_filters, kernel_size=kern1, activation='relu',padding='same')(b1)
    b1 = BatchNormalization()(b1)
    b2 = LSTM(lstm_filters,return_sequences=True)(input_layer)
    b2 = LSTM(lstm_filters,return_sequences=True)(b2)
    b2 = LSTM(lstm_filters,return_sequences=True)(b2)
    b3 = LSTM(lstm_filters,return_sequences=True)(input_layer)
    b3 = Conv1D(conv_filters, kernel_size=kern2, activation='relu',padding='same')(b3)
    b3 = BatchNormalization()(b3)
    b4 = MaxPooling1D(pool_size=3,padding='same',strides=1)(input_layer)
    b4 = SpatialDropout1D(0.10)(b4)
    concat1 = concatenate([b1,b2,b3,b4],axis=-1)
    return concat1

def inception_module(input_layer,conv_filters,kern1,kern2,kern3):
    b5 = Conv1D(conv_filters, kernel_size=kern1, activation='relu',padding='same')(input_layer)
    b6 = Conv1D(conv_filters, kernel_size=kern2, activation='relu',padding='same')(input_layer)
    b7 = MaxPooling1D(pool_size=3,padding='same',strides=1)(input_layer)
    b8 = Conv1D(conv_filters, kernel_size=kern3, activation='relu',padding='same')(input_layer)
    concat2 = concatenate([b5,b6,b7,b8],axis=-1)
    return concat2
#%% Defined SAR+MS model
def complex_fused_model():
    vh_input = Input(shape=(8,1))
    vh_conv_lstm = conv_lstm_module(vh_input,conv_filters=64,lstm_filters=64,kern1=3,kern2=3)
    vh_flt_conv_lstm = Flatten()(vh_conv_lstm)
    vh_inception = inception_module(vh_conv_lstm,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_inception = inception_module(vh_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vh_flt_inception = Flatten()(vh_flt_conv_lstm)

    vv_input = Input(shape=(8,1))
    vv_conv_lstm = conv_lstm_module(vv_input,conv_filters=64,lstm_filters=64,kern1=3,kern2=3)
    vv_flt_conv_lstm = Flatten()(vv_conv_lstm)
    vv_inception = inception_module(vv_conv_lstm,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_inception = inception_module(vv_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    vv_flt_inception = Flatten()(vv_inception)

    mss_input = Input(shape=(12,1))
    mss_conv_lstm = conv_lstm_module(mss_input,conv_filters=64,lstm_filters=64,kern1=3,kern2=3)
    mss_flt_conv_lstm = Flatten()(mss_conv_lstm)
    mss_inception = inception_module(mss_conv_lstm,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_inception = inception_module(mss_inception,conv_filters=64,kern1=3,kern2=5,kern3=7)
    mss_flt_inception = Flatten()(mss_inception)

    concat_inputs = concatenate([vh_flt_conv_lstm,vv_flt_conv_lstm,mss_flt_conv_lstm,
                                 vh_flt_inception,vv_flt_inception,mss_flt_inception])
    hidden1 = Dense(64, activation='relu')(concat_inputs)
    hidden2 = Dense(32, activation='relu')(hidden1)
    final_output = Dense(13, activation='softmax')(hidden2)
    final_model = Model(inputs=[vh_input,vv_input,mss_input],outputs=final_output)
    adam_optimizer = Adam(learning_rate=0.0001)
    final_model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return final_model
#%%
my_fused_model = complex_fused_model()
plot_model(my_fused_model, to_file='multiple_inputs.png', show_shapes=True,show_layer_names=False)
