import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,LeakyReLU,BatchNormalization
from keras import backend as K
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import numpy as np 


tf.compat.v1.disable_v2_behavior()
n_hidden_layers = 10 
dim_layer = 2 
data_dim = 1
activation = 'relu'#LeakyReLU(alpha=0.01)
def create_mlp_model(
    n_hidden_layers,
    dim_layer,
    input_shape,
    kernel_initializer,
    bias_initializer,
    activation,
):
    model = Sequential()
    model.add(Dense(dim_layer, input_shape=input_shape, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer))
    for i in range(n_hidden_layers):
        model.add(Dense(dim_layer, activation=activation, kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer))
    model.add(Dense(1, activation=activation, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer))
    return model

x_train = np.random.uniform(-np.sqrt(7),np.sqrt(7),(3000,1))
x_test = np.random.uniform(-np.sqrt(7),np.sqrt(7),(100,1))
y_train = np.abs(x_train)
dead_nn = 0
init = initializers.HeNormal()  
for i in range(100) :
    print(i)
    model = create_mlp_model(
        n_hidden_layers,
        dim_layer,
        (data_dim,),
        init,
        'zeros',
        activation)
    model.compile(loss="mean_squared_error", optimizer='adam')
    model.fit(x_train,y_train,epochs=1,batch_size=64,verbose=0)
    y_pred = model.predict(x_test)
    if np.var(y_pred)<=10**(-4) :
        dead_nn+=1
    
    

if __name__ == '__main__' :
    print(dead_nn/100)