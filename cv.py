import pandas as pd
import numpy as np
from lr import lr
from statistics import mean,stdev
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

#sample the data based on the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

#read the trainingSet for lr and svm
trainData = pd.read_csv('trainingSet.csv')

#sample the training sets
trainData = sampleData(trainData,18,1)

#split the data into 10 sets of equal sizes
LR_data_sets = np.array_split(trainData,10)

t_fracs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]

lr_test_accuracy = []

lr_test_accuracy_list = []

lr_stdev_list = []

for t_frac in t_fracs:
    for index in range(10):
  
        #copy the dataframes to get the sIndex and sC for each of the ten sets
        temp_LR=LR_data_sets.copy()
       
        #sIndex is the set of the values from training beloning to index
        setIndexLRSVM=temp_LR[index]
        del temp_LR[index]

        #sC, the remaining values other than sIndex
        setCLRSVM = pd.concat(temp_LR)

        
        #saample the data with random state 32 and the given t_frac
        train_set_LRSVM = sampleData(setCLRSVM,32,t_frac)
        
        #train lr and get the accuracy for test
        trainingAccuracy,testAccuracy = lr(train_set_LRSVM,setIndexLRSVM)
        lr_test_accuracy.append(testAccuracy)


    #calculate the mean of the 10 sets for each of the t_frac for lr
    lr_test_accuracy_list.append(mean(lr_test_accuracy))

    #calculate the standard error as the standard deviation of the ten sets for given t_frac divided by sqrt of number of sets
    lr_stdev_list.append(stdev(lr_test_accuracy)/math.sqrt(10))
    

    lr_test_accuracy.clear()

#get the training set size for each of the fraction to be plotted in x-axis of the graph
plot_x_axis = [t_frac * setCLRSVM.shape[0] for t_frac in t_fracs]



plt.errorbar( plot_x_axis, lr_test_accuracy_list, yerr= lr_stdev_list ,label='LR')

plt.xlabel('Training Dataset Size')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.title('Test Accuracy of LR and its standard errors')

plt.show()