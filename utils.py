from sklearn.model_selection import train_test_split
from sklearn import preprocessing,metrics
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% Function to label-encode, smote and split the data 
def preprocessing_split(parameters_list,train_data,test_data):
    train_y = train_data['crop']
    test_y = test_data['crop']
    #label encoding to assign int values to categorical labels
    le = preprocessing.LabelEncoder()
    train_y = le.fit_transform(train_y)
    y_test = le.fit_transform(test_y)
    train_X = train_data[parameters_list]
    test_X = test_data[parameters_list]
    train_X = train_X.to_numpy()
    X_test = test_X.to_numpy()

    sm = SMOTE(random_state=100)
    train_X,train_y = sm.fit_resample(train_X, train_y)

    np.random.seed(100)
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, train_size=0.75, random_state=1)
    return X_train,X_val,X_test,y_train,y_val,y_test

#%% Function to plot model's training and Validation accuracy and loss
def plot_performance(model,plot_title):
    plt.suptitle(plot_title)
    plt.subplot(1,2,1)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend(['Training','Validation'],loc='upper right')
    
    plt.subplot(1,2,2)
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Validation'],loc='lower right')
    plt.show()
    
#%% Function to return model metrics - Accuracy, Kappa, f1score, precision and recall
def model_metrics(model_predictions,ground_truth):    
    accuracy = metrics.accuracy_score(model_predictions,ground_truth)
    kappa = metrics.cohen_kappa_score(model_predictions,ground_truth)
    #print(str(model_name) + " accuracy with test data is " + str(accuracy))
    #print(str(model_name) + " kappa value with test data is " + str(kappa))
    table = (metrics.classification_report(model_predictions,ground_truth,digits=3,output_dict=True))
    scores = pd.DataFrame.from_dict(table)
    return [accuracy,kappa,scores]