#!/usr/bin/env python3
'''
Project TORQ for Computer vision course by Adam Qinawi
'''

import os
from cv2 import cv2
import matplotlib.pyplot as plt

def read_datasets():
    '''module reads all the datasets and returns them as 2d lists where
    each entry of main list is a list of [image data, label]

    Returns:
    -training_set: training dataset
    -validation_set: validation dataset
    -testing_set: testing dataset
    '''
    trainpath = 'TrainingSet/'
    validationpath = 'ValidationSet/'
    testpath = 'TestSet/'

    train = []
    validation = []
    test = []

    for label in os.listdir(trainpath):
        for image in os.listdir(trainpath + label):
            train.append((cv2.imread(trainpath + label +'/' + image),label))
    print("Training Set Loaded Successfully")

    for label in os.listdir(validationpath):
        for image in os.listdir(validationpath + label):
            validation.append((cv2.imread(validationpath + label +'/' + image),label))
    print("Validation Set Loaded Successfully")

    for label in os.listdir(testpath):
        for image in os.listdir(testpath + label):
            test.append((cv2.imread(testpath + label +'/' + image),label))
    print("Testing Set Loaded Successfully")

    return train,validation,test

def plot_accuracies(kvalues,kaccuracies):
    '''function to plot and display given accuracies

    parameters:
        -kvalues: list of k values used
        -kaccuracies: list of k accuracies observed

    returns None'''
    
    #Plot results in command line in case plot window does not open
    width = 40 
    print("K-value vs accuracy")
    for index, kvalue in enumerate(kvalues):
        accuracy = kaccuracies[index] * 100
        steps = int(accuracy*width/100) 
        print(f"k-value= {kvalue}|{'█'*steps}{' '*(width-steps)}|{accuracy:.2f}%")

    # this is the code for plotting accuracy through GUI
    # but it requires tkinter to be installed
    # I commented it out for the sake of avoiding needless errors

    # plt.plot(kvalues, kaccuracies, marker='o')
    # plt.xlabel('Values of K')
    # plt.ylabel('Accuracies observed')
    # plt.title('Effect of K-value on accuracy')
    # plt.show()


def test_across_k_values(model,training_set, kvalues):
    '''function to test model across different k-values

    parameters:
        -model: image classification model
        -testing_set: testing set to be used in the form ((image, label), (image, label)...)
        -kvalues: list of kvalues to be used

    returns:
        list of accuracies according to passed kvalues list'''
    accuracies = []
    for k in kvalues:
        print(f"testing with k-value= {k}")
        accuracy = model.test(training_set,k)
        accuracies.append(accuracy)
    return accuracies

def main():
    '''main script of program'''
    print(""" Adam's
███▀▀██▀▀███ ▄▄█▀▀██▄ ▀███▀▀▀██▄   ▄▄█▀▀██▄ 
█▀   ██   ▀███▀    ▀██▄ ██   ▀██ ▄██▀    ▀██▄
     ██    ██▀      ▀██ ██   ▄██ ██▀      ▀██
     ██    ██        ██ ███████  ██        ██
     ██    ██▄      ▄██ ██  ██▄  ██▄      ▄██
     ██    ▀██▄    ▄██▀ ██   ▀██▄▀██▄    ▄██▀
   ▄████▄    ▀▀████▀▀ ▄████▄ ▄███▄ ▀▀████▀▀  
                                       ███
                                        ▀████▀
Trained Object Recognition through kmeans Query\n\n""")
    #read datasets
    training_set, validation_set, testing_set = read_datasets()

    #instantiate model
    model = KNNModel(number_of_k_clusters=500) #number of clusters (words in vocabulary) is a hyperparameter
    #train model then display accuracy across k range
    model.train(training_set,load_vocab=False)
    k_range = range(1,10,2)
    accuracies = test_across_k_values(model, testing_set, k_range)
    plot_accuracies(k_range, accuracies)

if __name__ == "__main__":
    main()
