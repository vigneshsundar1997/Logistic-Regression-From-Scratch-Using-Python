from tkinter.filedialog import test
import pandas as pd
import numpy as np
from sys import argv
import warnings
warnings.filterwarnings("ignore")

#Split the input data into features set and decision set
def split_features_outcome(data):
    features = data.drop(['decision'],axis=1)
    decision = data['decision']
    return features,decision

#logistic regression model which takes trainingSet and testSet. Trains the model, calculates the accuracy of the trainingSet and the testSet and returns them.
def lr(trainingSet,testSet):
    #split the training dataset
    features,decision = split_features_outcome(trainingSet)
    shape = (1,features.shape[1]+1)
    #create an initial weight vector with zeros of size (1,261) i.e. (1,number of features+1)
    weight = np.zeros(shape)

    lambdaValue=0.01
    stepSize=0.01

    features_np_array = features.to_numpy()
    columns_shape = (features.shape[0],1)
    #add a column of ones to the feature array for first parameter in the weight vector
    feature_array = np.concatenate((np.ones(columns_shape),features_np_array),axis=1)
    decision_array = decision.to_numpy().reshape(-1,1)

    #iterate for 500 times
    for i in range(500):
        #initially calculate the dot product of feature and weight to apply in the sigmoid function
        weightTransposeFeature = np.dot(feature_array,weight.T)
        #calculate the value of the sigmoid function
        y_hat = 1 / ( 1 + np.exp(-weightTransposeFeature) )

        #calculate gradient as (y_hat - y)*X + lambda * weight
        decisionMinusYhatDotFeature = np.dot((y_hat - decision_array).T,feature_array)
        delta = decisionMinusYhatDotFeature + lambdaValue * weight

        #calcuate the new weight as old weight - step_size * delta
        new_weight = weight - (stepSize * delta)

        #if the new weight - weight is less than 1e-6, then we break the iterations and the new weight will be the learned weight.
        if(np.linalg.norm(new_weight-weight) < 1e-6):
            break
        else:
            weight=new_weight

    #predict the outcomes in training set using the new learned weight
    weightTransposeFeature = np.dot(feature_array,new_weight.T)
    y_hat = 1 / ( 1 + np.exp(-weightTransposeFeature) )


    trueCount=0
    # if the predicted value is less greater or equal to 0.5, label it is as 1, else label it as 0
    for index in range(len(decision_array)):
        if y_hat[index]>=0.5 and decision_array[index]==1:
            trueCount+=1
        elif y_hat[index]<0.5 and decision_array[index]==0:
            trueCount+=1

    
    training_accuracy = round(trueCount/len(decision_array),2)
    
    #predict the outcomes for the test set
    features,decision = split_features_outcome(testSet)
    features_np_array = features.to_numpy()
    columns_shape = (features.shape[0],1)
    feature_array = np.concatenate((np.ones(columns_shape),features_np_array),axis=1)
    decision_array = decision.to_numpy().reshape(-1,1)

    weightTransposeFeature = np.dot(feature_array,new_weight.T)
    y_hat = 1 / ( 1 + np.exp(-weightTransposeFeature) )

    trueCount=0

    for index in range(len(decision_array)):
        if y_hat[index]>=0.5 and decision_array[index]==1:
            trueCount+=1
        elif y_hat[index]<0.5 and decision_array[index]==0:
            trueCount+=1
    test_accuracy = round(trueCount/len(decision_array),2)

    return training_accuracy,test_accuracy

if __name__ == "__main__":
    trainingDataFileName = argv[1]
    testDataFileName = argv[2]

    data_train=pd.read_csv(trainingDataFileName)
    data_test=pd.read_csv(testDataFileName)
    training_accuracy,test_accuracy=lr(data_train,data_test)
    print('Training Accuracy LR:', training_accuracy)
    print('Testing Accuracy LR:', test_accuracy)