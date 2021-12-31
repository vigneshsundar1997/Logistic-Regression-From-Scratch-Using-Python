# Logistic-Regression-From-Scratch-Using-Python
Implementation of Logistic Regression model from scratch using Python on Dating dataset

The project folder contains 3 python files: 
1. preprocess-assg3.py
2. lr.py
3. cv.py

#####
1. preprocess-assg3.py

This script contains the preprocessing steps like removing the quotes, converting to lower case, normalization, one-hot encoding and split the dataset. It makes use of dating-full.csv as the input. It outputs trainingSet.csv and testSet.csv.

Execution : python3 preprocess-assg3.py

2. lr.py

This script contains the training and testing of the model for Logistic Regression. It takes in two arguments, the training file name, the test file name.

Execution : python3 lr_svm.py trainingFileName testFileName

eg: 
Run command for LR model
python3 lr_svm.py trainingSet.csv testSet.csv.  

3. cv.py

This script performs the ten fold cross validation for the LR model. It also outputs a graph indicating the test accuracies of the model and their standard errors for different trainingSet sizes.

This takes the trainingSet size as the hyper parameter and performs the CV.

Note: 
1. Once the script is run, it takes some time to train and test the models for each training size and each index. After sometime, it outputs the graph and the t-test statistics.
2. Before running this script, run the preprocess-NBC.py to produce the files needed for training and testing NBC model.
Execution : python3 cv.py
