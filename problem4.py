import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,LeakyReLU
from keras import backend as K
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split

def f(x) :
    x1=x[:,0]
    x2=x[:,1]
    return -(x2+47)*np.sin(np.sqrt(np.abs(x1/2+x2+47)))-x1*np.sin(np.sqrt(np.abs(x1-x2-47)))
x=np.random.uniform(-512,512,[100000,2])
y=f(x)+np.random.normal(0,0.3)
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.2)

size_list = [16,32,64,128,256,512]
def architect() :
    res1=[]
    res2=[]
    res3=[]
    for i in range(6) :
        res1.append([size_list[i]])
    for j in range(6) :
        for i in range(len(res1)) :
            if np.sum(res1[i])+size_list[j]<= 512 :
                res2.append(res1[i]+[size_list[j]]) 
    for k in range(6) :
        for j in range(len(res2)) :
            if np.sum(res2[j])+size_list[k]<=512 :
                res3.append(res2[j]+[size_list[k]]) 
    return res1,res2,res3

def create_model(neuron_list) :
    model=Sequential()
    model.add(Dense(n,input_shape = 2,activation='relu',kernel_initializer = initializers.HeNormal(),biaz_initilizer = 'zeros'))
    for i in range(len(neuron_list)-1) :
        model.add(Dense(neuron_list[i+1],input_shape = neuron_list[i],activation='relu',kernel_initializer = initializers.HeNormal(),biaz_initilizer = 'zeros'))
    model.add(Dense(1,activation='relu'))
    return model 

def get_params(neuron_list) :
    res=neuron_list[0]*2
    for i in range(len(neuron_list)-1) :
        res+=neuron_list[i]*neuron_list[i+1]
    return res+neuron_list[-1]
list1,list2,list3 = architect()

loss1={}
lossw1={}
for neuron_list in list1 :
    model = create_model(neuron_list)
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
    model.compile(loss="mean_square_error",optimizer=optimizer)
    model.fit(x_train,y_train,epochs=2000,batch_size=1000)
    y_pred = model.predict(x_test)
    loss=np.sqrt(keras.losses.mean_square_error(y_test,y_pred))
    loss1[neuron_list[0]] = loss
    loss1w[get_params(neuron_list)]

loss2={}
lossw2={}
for neuron_list in list2 :
    model = create_model(neuron_list)
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
    model.compile(loss="mean_square_error",optimizer=optimizer)
    model.fit(x_train,y_train,epochs=2000,batch_size=1000)
    y_pred = model.predict(x_test)
    loss=np.sqrt(keras.losses.mean_square_error(y_test,y_pred))
    if np.sum(neuron_list) in loss2 :
        loss2[np.sum(neuron_list)].append(loss)
    else : 
        loss2[np.sum(neuron_list)] = [loss]  
    nb_param = get_params(neuron_list)  
    if nb_param in lossw2 :
        lossw2[np.sum(nb_param)].append(loss)
    else : 
        lossw2[np.sum(nb_param)] = [loss]
for key in loss2 :
    loss2[key] = np.mean(loss2[key])
for key in lossw2 :
    lossw2[key] = np.mean(lossw2[key])

loss3={}
lossw3={}
for neuron_list in list3 :
    model = create_model(neuron_list)
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=True)
    model.compile(loss="mean_square_error",optimizer=optimizer)
    model.fit(x_train,y_train,epochs=2000,batch_size=1000)
    y_pred = model.predict(x_test)
    loss=np.sqrt(keras.losses.mean_square_error(y_test,y_pred))
    if np.sum(neuron_list) in loss3 :
        loss3[np.sum(neuron_list)].append(loss)
    else : 
        loss3[np.sum(neuron_list)] = [loss]
    nb_param = get_params(neuron_list)  
    if nb_param in lossw3 :
        lossw3[np.sum(nb_param)].append(loss)
    else : 
        lossw3[np.sum(nb_param)] = [loss]
for key in loss3 :
    loss3[key] = np.mean(loss3[key])
for key in lossw3 :
    lossw3[key] = np.mean(lossw3[key])








if __name__ == "__main__" :
    print(get_params([2,2,2]))
    
