import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import seed
seed(100)
tf.random.set_seed(100)

tf.config.run_functions_eagerly(True)
#os.environ['PROJ_LIB'] = "D:/anaconda3/envs/gpu-enabled-spatial/Library/share/proj"
physical_devices = tf.config.list_physical_devices("GPU")
#print("Num GPUs Available: ", len(physical_devices))
#%%
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res  
#%%
from keras.layers.merge import concatenate
from keras import Model
from tensorflow.keras.optimizers import Adam

#n_classes = len(np.unique(y_train))
def fusion_transformer(head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,dropout=0,mlp_dropout=0):
    input1 = keras.Input(shape=(8,1))
    x = input1
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    output1 = layers.Dense(64, activation="relu")(x)
    
    input2 = keras.Input(shape=(8,1))
    x = input2
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    output2 = layers.Dense(64, activation="relu")(x)

    input3 = keras.Input(shape=(12,1))
    x = input3
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    output3 = layers.Dense(64, activation="relu")(x)

    final_concat = concatenate([output1,output2,output3],axis=-1)
    hidden1 = layers.Dense(64, activation='relu')(final_concat)
    hidden2 = layers.Dense(32, activation='relu')(hidden1)
    hidden3 = layers.Dense(16, activation='relu')(hidden2)
    final_output = layers.Dense(13, activation='softmax')(hidden3)
    final_model = Model(inputs=[input1,input2,input3],outputs=final_output)
    #adam_optimizer = Adam(learning_rate=0.0001)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0001,decay_steps=100000,decay_rate=0.96,staircase=True)
    final_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy'])
    
    return final_model
    #return keras.Model(inputs, outputs)
