import csv
import numpy as np
import tensorflow as tf
#tf.python.control_flow_ops = tf
np.random.seed(1337)  # for reproducibility
from sklearn import svm
from sklearn import cross_validation
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from nolearn.dbn import DBN
from sklearn.metrics import classification_report, accuracy_score
#from dbn.tensorflow import SupervisedDBNClassification
import matplotlib.pyplot as plt
from keras.models import load_model

def remove_half(mat):
    
#    length = mat.shape[0]
    
    for i in range(0, 20):
        
        max_mat = mat.argmax(axis = 0)
        mat[max_mat] = 0
        
    return mat


def combine_9_layers(w, new8):
#    w0 = w[0]
#    new_w = w0[np.newaxis, :]
#    
#    for j in range(0, 7):
#        w_tem = w[j + 1]
#        w_tem = w_tem[np.newaxis, :]
#        print new_w.shape, w_tem.shape
#        new_w = np.append(new_w, w_tem, axis = 0)
#        
#    new8 = new8[np.newaxis, :]
#    new_w = np.append(new_w, new8, axis = 0)
    
    lis = [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], new8]
    
    return lis


        

save_new_w_of_part1 = np.zeros((9, 112),dtype = float)
f, plots = plt.subplots(2, 2, figsize=(100, 100))

for step in range(0, 3):
    
    features= []
    label = []

    features2 = []
    label2 = []

    with open('/Users/wuzhenglin/Python_nice/SAL_EGG/eeg-brain-wave-for-confusion/EEG_data.csv','rb') as f:
        reader = csv.reader(f)
        i=0
        for row in reader:
            if(i==0):
                i+=1
                continue
            for i in range(0,len(row)):
                row[i] = float(row[i])
            
            features.append(row[0:14])
            label.append(row[14])
            
            features2.append(row[0:14])
            label2.append(row[0])
        

        
    if step == 0:
        print '############ONE###############'
        X = np.asarray(features)
        Y = np.asarray(label)
    
    if step == 1:
        print '############SECOND###############'
        X = np.asarray(features2)
        Y = np.asarray(label2)
        X_ = np.asarray(features)
        Y_ = np.asarray(label)
        
    if step == 2:
        print '############THIRD###############'
        X = np.asarray(features)
        Y = np.asarray(label)
    
    features = {}
    output = {}
    

    for i in range(X.shape[0]):
        tu = int(X[i][0]*10 + X[i][1])
        
        if tu not in features.keys():
            features[tu] = X[i][2:14]
        elif features[tu].shape[0] < 1344:
            features[tu] = np.concatenate((features[tu],X[i][2:14]),axis =0)
        
        output[tu]= Y[i]
    
    input = np.zeros((100,1344),dtype = float)
    labels = np.zeros((100,1),dtype = int)
    
    for i in features.keys():
        
        input[i,:] = features[i]
        labels[i] = output[i]
    
    
    print "Begin LSTM model"
    accuracy = 0.0
    
    weight_final_sum = np.zeros((112),dtype = float)
    weight_finial_all = [[]]
    
    
    if step == 0:
        num = 5
    else:
        num = 5
        
    for i in range(0, num):
        print 'The EPOCH', i
        X_train, X_test, Y_train, Y_test = train_test_split(input, labels, test_size=0.2, random_state=i*10)
        
        X_train = X_train.reshape(80,112,12)
        X_test = X_test.reshape(20,112,12)
        
        y_train = np.zeros((80,112),dtype='int')
        y_test = np.zeros((20,112),dtype='int')
        
        y_train = np.repeat(Y_train,112, axis=1)
        y_test = np.repeat(Y_test,112, axis=1)
        np.random.seed(1)
        
        
        
        # create the model
        model = Sequential()
        batch_size = 20
    
        model.add(BatchNormalization(input_shape=(112,12), mode = 0, axis = 2))#4
        model.add(LSTM(100, return_sequences=False, input_shape=(112,12))) #7 

        model.add(Dense(112, activation='hard_sigmoid'))#9
        model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['binary_accuracy'])#9
        
        if step == 1:
            
            del model
            model = load_model('LSTM_first_train_model.h5')
            print 'load model1111111111'
            model.fit(X_train, y_train, nb_epoch=30)#9
            wei1 = model.get_weights() 
            print wei1[8]
        
        if step == 2:
            
            del model
            model = load_model('LSTM_second_train_model.h5')
            print 'load model22222222222222'
            wei2 = model.get_weights() 
            print wei2[8]
            

    
        if step == 0:
            
            model.fit(X_train, y_train, nb_epoch=30)#9
            model.save('LSTM_first_train_model.h5')
            print 'save model111111111'
            wei0 = model.get_weights() 
            print wei0[8]
        
        if step != 2:
            
            w = model.get_weights()          
            weight_final = w[8]        
            
            #imshow all epochs
            weight_final_op = weight_final
            weight_final_op = weight_final_op[np.newaxis, :]
            
            if i == 0:
                
                weight_finial_all = weight_final_op
            
            else:
                           
                weight_finial_all = np.append(weight_finial_all, weight_final_op, axis = 0)
            
            if step == 1 and i == 4:
                
                #cal sun in all epochs and create the new weigh of w
                weight_final_sum = weight_final_sum + weight_final        
                new_weight_final = remove_half(weight_final_sum)       
                

                
                del model
                model = load_model('LSTM_first_train_model.h5')
                
                wnew = model.get_weights()
                print wnew[8]
                new_w = combine_9_layers(wnew, new_weight_final)        
                model.set_weights(new_w)
                print 'new w'
                model.save('LSTM_second_train_model.h5')
                print 'save model222222222'
                
                
            # Final evaluation of the model
            scores = model.evaluate(X_test, y_test, batch_size = batch_size, verbose=0)
    
            
            
    
            
            if step == 0 and i == 4:
                plots[0, 0].axis('off')
                plots[0, 0].imshow(weight_finial_all, cmap = plt.cm.YlGn)
                
            if step == 1 and i == 4:
                plots[0, 1].axis('off')
                plots[0, 1].imshow(weight_finial_all, cmap = plt.cm.Blues)
                
            print("Accuracy: %.2f%%" % (scores[1]*100))
         
        else:
            
            scores = model.evaluate(X_test, y_test, batch_size = batch_size, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1]*100))
            break
        
      
        
        
        

        accuracy += scores[1]
    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$AVG$$$$$$$$$$$$$$$$$$$$$$$$$$$'     
    print accuracy/5


#average accuracy: 0.690000013262