import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import random


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)




def CNN(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=10, batch_size=32,verbose=0)
    true=[0,0,0,0,0,0,0,0,0,0]
    accuracy=[0,0,0,0,0,0,0,0,0,0]

    for i in range(len(x_test)):
        y_pre = model.predict(x_test[i].reshape(1,28,28,1))
        test = int(y_test[i])
        #print(test)
        true[test] += 1
        for y in range(len(y_pre[0])):
            accuracy[y]+=y_pre[0][y] 
        #print("accuracy = ", accuracy,"y_pred = ", y_pre, "true =", true)
    return accuracy, true



def RandomSample(X_train, Y_train, X_test,Y_test):
    perc = 0.1
    for i in range(100):
        filename = 'sample'
        filename = filename+str(i)
        x_train =[]
        y_train = []
        if i%10==0:
            perc+=0.1

        for i in range(0,int(perc*X_train.shape[0])):
            Rand = random.randint(0, X_train.shape[0]-1)
            x_train.append(X_train[Rand])
            y_train.append(Y_train[Rand])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        accuracy,true =CNN(x_train,y_train,X_test,Y_test)
        SaveToFile(filename,x_train,y_train,accuracy, true)



def SaveToFile(filename, x_train, y_train, accuracy, true):
    data = open(filename+'.txt','w')
    result = open(filename+'res.txt','w')

    for i in range(len(x_train)):
        data.write(str(x_train[i]))
    for i in range(len(y_train)):
        data.write(str(y_train[i])+'\n')
    for i in range(len(accuracy)):
        #print("done Writing")
        result.write("accuracy = "+str(accuracy[i])+" true = " +str(true[i])+'\n')
    data.close()
    result.close()
        


RandomSample(X_train, Y_train, X_test, Y_test)



