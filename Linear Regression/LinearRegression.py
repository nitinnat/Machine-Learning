
# coding: utf-8

"""
Version: Python 3.5


"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
random.seed(12345)


"""
read_data() takes a filepath and returns a dataframe with the data.
"""
def read_data(filepath):
    df = pd.read_csv(filepath)
    return np.array(df)



"""
The function createSplits() takes in the input data and splits
it into training, testing and validation sets. 

Inputs:
data = Input dataset to be split.
labels = The output labels to be split
ratios = an array of size 1 x 3, representing the ratios of train, 
         validation and test respectively.
randomize = True by default. If True, shuffle the input data set once,
            before doing the splits.
            
Outputs:
dataset = a dictionary containing X_train, X_valid, X_test, y_train, y_valid and y_test.

"""

def createSplits(data, labels, ratios = [0.8, 0.1, 0.1], randomize = True):
    #Check if the sizes of both arrays match
    assert len(data) == len(labels), "The number of data points must be equal to the number of labels."
    if sum(ratios) != 1:
        print("The partition ratios must sum to 1.")
        return -1
    if randomize:
        #We randomize the data and labels together
        temp = list(zip(data,labels))
        random.shuffle(temp)
        data, labels = map(list,zip(*temp))
    data_size = len(data)
    #Store all the arrays in a dictionary. Makes it easy to return.
    dataset = {}
    dataset["X_train"] = np.array(data[0:round(ratios[0]*data_size)])
    print("Train set size: "+ str(len(dataset["X_train"])))
    dataset["X_valid"] = np.array(data[round(ratios[0]*data_size):round((ratios[0] + ratios[1])*data_size)])
    print("Validation set size: "+ str(len(dataset["X_valid"])))
    dataset["X_test"] =  np.array(data[round((ratios[0] + ratios[1])*data_size):])
    print("Test set size: "+ str(len(dataset["X_test"])))
    dataset["y_train"] = np.array(labels[0:round(ratios[0]*data_size)])
    dataset["y_valid"] = np.array(labels[round(ratios[0]*data_size):round((ratios[0] + ratios[1])*data_size)])
    dataset["y_test"] = np.array(labels[round((ratios[0] + ratios[1])*data_size):])
    return dataset


"""
The function getClusterIndices() takes a cluster label and returns all the 
indices in the dataset corresponding to that cluster label.

Inputs: 
clustNum = the cluster for which you need the indices.
labels_array = The array of output labels.

Outputs:
A list of indices from the data set corresponding to a cluster label.
"""

def getClusterIndices(clustNum, labels_array): #list comprehension
    a = np.array([i for i, x in enumerate(labels_array) if x == clustNum])
    return np.reshape(a,[1,len(a)])

"""
The function chooseCenters() takes the dataset and number of clusters as an input,
and returns X_split, centers and spreads.

Inputs:
X_split = contains a list of M arrays. Each array contains all the training data points that
          were assigned that cluster.
centers = An array of M arrays, where each array is of size 1 x 46. These are the centroids of
          each cluster.
          
spreads = An array of M arrays. Each array is a diagonal matrix of size 46 x 46, and is calculated
          by taking the dimensional variances across 46 dimensions, within each cluster. The vector is
          then diagonalized.
"""

def chooseCenters(X,M):
    km = KMeans(n_clusters=M).fit(X)
    centers = km.cluster_centers_
    
    centers = centers[:, np.newaxis, :]
    labels = km.labels_
    X_split = np.array([X[getClusterIndices(i,labels)[0]] for i in range(0,M)]) #Contains M arrays
    X_split = np.reshape(X_split,[M, -1])

    variance_vec = X.var(axis=0)/10
    spreads = np.array([np.diag(variance_vec) for i in range(M)])
    #Calculate spreads for each cluster. Will be a M 1 x 46 dimensional vectors
    #spreads = np.array([X_split[i][0].var(axis=0)/10 for i in range(len(X_split))])
    
    #Diagonalization. Convert the spreads into a diagonal form.
    #For each spread of 46 x 1 dimensions, convert it into a diagonal matrix.
    #spreads = np.array([np.diag(spreads[i]) for i in range(len(spreads))])
    
    return X_split, centers, spreads


"""
The function generateDesignMatrix() takes in input dataset X,
the centers and the spreads obtained by from each of the M+1 clusters.

Inputs:
X = Input dataset of size N x M
centers = The centers obtained from the Kmeans analysis.
spreads = The spreads obtained from the Kmeans analysis.

Outputs:
Design Matrix of size N x M+1
"""

def generateDesignMatrix(X,centers,spreads):
    X = X[np.newaxis, :, :]
    
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers),axis=2) / (-2)).T
    
    return np.insert(basis_func_outputs, 0, 1, axis=1)



"""
This function outputs the closed form solution for a given design matrix 
and a set of output labels.

Inputs: 
design_matrix = A matrix of N x (M+1) dimensions, where N is the number of input samples,
                and M is the number of basis functions selected. It is M+1 because of the 
                bias terms inserted into the first column.
labels = This is an array of N x 1 dimensions representing your output labels.
reg_lambda = Lambda value/Lagrangian coefficient used for L2 regularization. Set to 0.1 by default.
             Set to 0 for no regularization.

Outputs:
Returns the closed form weights of the solution to the Linear Regression solution.

"""


def closedFormSolution(design_matrix,labels, reg_lambda = 0.1):
    return np.linalg.solve(
        reg_lambda * np.identity(design_matrix.shape[1]) +
        np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T, labels)
    ).flatten()


"""
Evaluates the Sum of Squared differences error given weights, design matrix Phi and target t

"""
def evaluate_error(weights,Phi, t):
    weights = np.reshape(weights, [1,-1])
    E_D = (1/2)* np.sum(np.power((np.matmul(Phi, weights.T)-t),2))
    return E_D

"""

Calculates the RMSE, given the data-dependent error E and size of data set N
"""
def calcERMS(E,N): return pow(2*E/N,0.5)




"""
The function SGD_sol() returns the weights and errors obtained after stochastic gradient descent.
Inputs: learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_labels
Outputs: w_sgd (Weights obtained), errors = A list of ERMS values after each minibatch of training

"""

def SGD_sol(learning_rate,minibatch_size,num_epochs,
            L2_lambda,
            design_matrix,
            output_data,
            weights):
    errors = []
    N, M = design_matrix.shape
    #weights = np.random.randn(1,M)*0.01#np.zeros([1, design_matrix.shape[1]])
    for epoch in range(num_epochs):
        for i in range(round(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul(
                (np.matmul(Phi, weights.T)-t).T,
                Phi
            )
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
            
        errors.append(calcERMS(np.linalg.norm(E),N))
        
    return weights.flatten(),errors



"""
The grid_search() function performs a grid search over hyperparameters
and returns the best values obtained after checking the error on the validation set.

Inputs:
X_path, y_path = Paths to the respective training data and training labels.

Outputs:
best_M              = Best value of M
best_L2_lambda      = Best value of Regularization parameter Lambda
best_learning_rate  = Best value of learning rate
best_valid_error    = Best value of validation error
best_test_error     = Best value of test error
best_weights_sgd    = Best value of weights for SGD
grid_search_dict    = A dictionary containing all the results obtained in the grid search.

"""


def grid_search(X_path,y_path):
    Ms = [5,6,7,8]
    learning_rates = [0.05,0.08,0.1]
    L2_lambdas = [0.5,0.6]

    best_valid_error = float("inf")
    grid_search_dict = {}
    count = 0
    for i, M_val in enumerate(Ms):
        for j,L2_lambda_val in enumerate(L2_lambdas):
            for k,learning_rate_val in enumerate(learning_rates):
                print("Running SGD...")
                print("L2_Lambda: " + str(L2_lambda_val))
                print("learning rate: " + str(learning_rate_val))
                print("M: " + str(M_val+1))
                w_cf, w_sgd, validation_error, test_error, errors = LinearRegression(X_path, 
                             y_path,
                             M=M_val, 
                             learning_rate = learning_rate_val,
                             L2_lambda = L2_lambda_val, 
                             num_epochs = 5000, 
                             minibatch_size = 512
                            )
                grid_search_dict[count] = {}
                grid_search_dict[count]['M'] = M_val
                grid_search_dict[count]['L2_Lambda'] = L2_lambda_val
                grid_search_dict[count]['learning_rate'] = learning_rate_val
                grid_search_dict[count]['errors'] = errors
                grid_search_dict[count]['validation_error'] = validation_error
                grid_search_dict[count]['test_error'] = test_error
                grid_search_dict[count]['weights'] = w_sgd
            
                if validation_error < best_valid_error:
                    best_valid_error = validation_error
                    best_test_error = test_error
                    best_weights_sgd = w_sgd
                    best_learning_rate = learning_rate_val
                    best_M = M_val
                    best_L2_lambda = L2_lambda_val
                count += 1
    return best_M, best_L2_lambda, best_learning_rate,best_valid_error, best_test_error, best_weights_sgd, grid_search_dict


"""

This function plots the results obtained by teh grid search.

"""

def plot(grid_search_dict):
    #num_plots = int(pow(len(grid_search_dict.keys()),0.5)) + 1
    cols = 3
    rows = int(len(grid_search_dict.keys())/3) + 1
    fig, axs = plt.subplots(rows,cols,figsize=(30,30))
    
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= len(grid_search_dict.keys()):
                break
            else:
                err = grid_search_dict[count]['errors']
                axs[i,j].plot(range(len(err)), err)
                title = ("Learning rate: " + str(grid_search_dict[count]['learning_rate']) +
                         " Lambda: " + str(grid_search_dict[count]["L2_Lambda"]) +
                         " M: " + str(grid_search_dict[count]['M']))
                axs[i,j].set_title(title)
                    
                count = count + 1


#Driver code for Linear Regression
"""
This is the main function. 
Inputs:
X_path, y_path = File paths to the training data and training labels respectively.
M = Number of basis functions - 1. NOTE: For this implementation, the design matrix is of M+1 size due to the added bias.
learning_rate
L2_lambda = Regularization parameter
num_epochs = Number of iterations to run the algorithm
minibatch_size = Size of each batch after which an update to the weights is made.

Outputs:

w_cf = Closed form weights
w_sgd = SGD weights
validation_error_rms = RMSE error on the validation set
test_error_rms = RMSE error on the test set
len(X_train), len(X_valid), len(X_test) = Lengths of respective datasets. Used later in the grid search function
errors = List of RMSE errors obtained after each batch of training

"""
def LinearRegression(X_path, 
                     y_path,
                     M, 
                     learning_rate = 0.15,
                     L2_lambda = 0.5, 
                     num_epochs = 10, 
                     minibatch_size = 20
                    ):
    
    #Input Data
    letor_input_data = read_data(X_path)
    letor_output_data = read_data(y_path)
    
    #Create Splits
    split_dic = createSplits(letor_input_data,letor_output_data)    
    X_train = split_dic['X_train']
    y_train = split_dic['y_train']
    X_valid = split_dic['X_valid']
    y_valid = split_dic['y_valid']
    X_test = split_dic['X_test']
    y_test = split_dic['y_test']
    
    
    #Run Kmeans and create basis functions
    X_split, centers, spreads = chooseCenters(X_train,M)
    
    
    #Generate design matrices for train, validation and test datasets
    design_matrix_train = generateDesignMatrix(X_train, centers, spreads)
    design_matrix_valid = generateDesignMatrix(X_valid, centers, spreads)
    design_matrix_test = generateDesignMatrix(X_test, centers, spreads)
    
    #Reassign M after inserting phi0
    M = design_matrix_train.shape[1]
    
    #Obtain closed form solution
    w_cf = closedFormSolution(design_matrix_train,y_train,L2_lambda)
    validation_error_cf = evaluate_error(w_cf,design_matrix_valid,y_valid)
    test_error_cf = evaluate_error(w_cf,design_matrix_test,y_test)
    
    #Obtain SGD solution using Early Stopping method
    w_sgd = np.random.randn(1,M)*0.01
    print("Creating a random weights matrix of size: " + str(M))
    Nv = design_matrix_valid.shape[0]
    #Setting patience parameter
    p = 1
    j = 0
    i = 0 
    v = float("inf")
    while j<p:
        print(j+1)
        w_sgd,errors =  SGD_sol(learning_rate,
                            minibatch_size,
                            num_epochs,
                            L2_lambda,
                            design_matrix_train,
                            y_train,
                            w_sgd)
        print(w_sgd.shape)
        i = i + num_epochs
        v1 = evaluate_error(w_sgd,design_matrix_valid,y_valid )
        vrms = calcERMS(v1,Nv)
        print(v1, vrms)
        if v1 < v:
            print("Validation error is decreasing...Optimizing some more")
            j = 0
            w_star = w_sgd
            
            i_star = i
            v = v1
            vrms = calcERMS(v1,Nv)
        else:
            print("Validation error is not decreasing. Time step: " + str(j+1) + ". Trying again...")
            j += 1
    
    
    w_sgd = w_star
    
    validation_error = evaluate_error(Phi = design_matrix_valid,t = y_valid,weights=w_sgd)
    validation_error_rms = calcERMS(validation_error,len(y_valid))
    test_error = evaluate_error(w_sgd,design_matrix_test,y_test)
    test_error_rms = calcERMS(test_error,len(y_test))
    return w_cf,w_sgd, validation_error_rms, test_error_rms, errors, validation_error_cf, test_error_cf
    


def main():

    X_path_LETOR = "Querylevelnorm_X.csv"
    y_path_LETOR = "Querylevelnorm_t.csv"
    X_path_syn = "input.csv"
    y_path_syn = "output.csv"
    """
    #Run grid search function
    best_M, best_L2_lambda, best_learning_rate, best_valid_error, best_test_error, w_sgd, grid_search_dict = grid_search(X_path,y_path)
    print("Best M: " + str(best_M))
    print("Best Lambda: "+ str(best_L2_lambda))
    print("Best Learning Rate: " + str(best_learning_rate))
    print("Best validation error: " + str(best_valid_error))
    print("Best test error: " + str(best_test_error))

    plot(grid_search_dict)        
    """
    print("Running Linear Regression on the LETOR dataset...")
    (w_cf_LETOR, w_sgd_LETOR, valid_error_LETOR, 
    test_error_LETOR, errors_LETOR, 
    valid_error_cf_LETOR, test_error_cf_LETOR) = LinearRegression(X_path_LETOR,y_path_LETOR,
	   															    M=5, 
		  														    learning_rate = 0.1,
																    L2_lambda = 0.5, 
                                                                    num_epochs = 10, 
																    minibatch_size = 1
																    )

    print("Running Linear Regression on the Synthetic dataset...")
    (w_cf_syn, w_sgd_syn, valid_error_syn, 
    test_error_syn, errors_syn,
    valid_error_cf_syn, test_error_cf_syn) = LinearRegression(X_path_LETOR,y_path_LETOR,
																    M=6, 
																    learning_rate = 0.01,
																    L2_lambda = 0.5, 
																    num_epochs = 10, 
																    minibatch_size = 1
																    )
    print("Weights of closed form solution on LETOR dataset: " + str(w_cf_LETOR))
    print("Weights of closed form solution on synthetic dataset: " + str(w_cf_syn))

    print("Validation error of SGD solution on LETOR is " + str(valid_error_LETOR))
    print("Test error on of SGD solution on LETOR is " + str(test_error_LETOR))
    print("Validation error of SGD solution on synthetic is " + str(valid_error_syn))
    print("Test error of SGD solution on synthetic is " + str(test_error_syn))




