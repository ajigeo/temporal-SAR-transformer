from sklearn.model_selection import train_test_split
from sklearn import preprocessing,metrics
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

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

#%% Extracting labels
def get_crop_classes(training_data_1st_column):
    le = preprocessing.LabelEncoder()
    le.fit_transform(training_data_1st_column)
    labels = le.classes_
    return labels

#%% Function to plot confusion matrix
def confusion_matrix(labels,ground_truth,model_predictions,model_name):
    plt.figure(figsize=(7,7))
    ax = plt.subplot()
    matrix = metrics.confusion_matrix(ground_truth,model_predictions)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sb.heatmap(matrix,annot=True,linewidths=.5,fmt=".2f",cbar=False,cmap='Greens') #integer dtype
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    ax.xaxis.set_ticklabels(labels,rotation=45)
    ax.yaxis.set_ticklabels(labels,rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title(model_name)
    plt.show()
