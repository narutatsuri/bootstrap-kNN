from sklearn.datasets import load_breast_cancer, load_wine, fetch_covtype, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from Bootstrap_kNN import Bootstrap_kNN
import numpy as np
from util import *
import random


def make_dataset():
    """ 
    Loads raw train and test datasets and formats them. 
    INPUTS:     None
    RETURNS:    Train and Test datasets
    """
    print("Using ", dataset_name, " dataset.")
    if dataset_name == "breast_cancer":
        dataset = load_breast_cancer()
    elif dataset_name == "wine":
        dataset = load_wine()
    elif dataset_name == "covertype":
        dataset = fetch_covtype()
    elif dataset_name == "iris":
        dataset = load_iris()
        
    # Split dataset into train and test sets
    return train_test_split(dataset.data,
                            dataset.target, 
                            random_state=1)

def column(matrix, i):
    return [row[i] for row in matrix]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = make_dataset()
    
    #? k Nearest Neighbors
    kNN = KNeighborsClassifier(n_neighbors=k)
    # Fit kNN on training data
    kNN.fit(X_train, y_train)
    # Get accuracy
    print("kNN Accuracy: ", kNN.score(X_test, y_test) * 100, "%")
    
    #? Bootstrap-k Nearest Neighbors
    BkNN = Bootstrap_kNN(k)
    ystar_test = BkNN.train([X_train, y_train], [X_test, y_test], epochs)
            