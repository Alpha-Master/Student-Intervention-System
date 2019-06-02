# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:26:55 2018

@author: kuttattu
"""

# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

# TODO: Import the three supervised learning models from sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Read student data
student_data = pd.read_csv("student-data.csv")
print("Student data read successfully!")

# TODO: Calculate number of students
n_students = len(student_data)

# TODO: Calculate number of features
n_features = len(student_data.columns) - 1 # 30 feature columns, one target column

# TODO: Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])

# TODO: Calculate failing students
n_failed = n_students - n_passed

# TODO: Calculate graduation rate
grad_rate = float(n_passed) / float(n_students) * 100

# Print the results
print("Total number of students: {}".format(n_students))
print("Number of features: {}".format(n_features))
print("Number of students who passed: {}".format(n_passed))
print("Number of students who failed: {}".format(n_failed))
print("Graduation rate of the class: {:.2f}%".format(grad_rate))

# Extract feature columns headers
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    #df.iteritems(): Iterator over (column name, Series) pairs.
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''  
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))
        
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''   
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()    
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    confusion_matric = confusion_matrix(target, y_pred)
    print("Confusion matrix:")
    print(confusion_matric)
    print("Classification_report:")
    print(classification_report(target, y_pred))
    return f1_score(target.values, y_pred, pos_label=1)
    

def train_predict(clf, X_train, y_train, X_test, y_test):

    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size

    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print ("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

X_all = preprocess_features(X_all)

print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
#Change target label from string to int
y_all = y_all.replace(['yes', 'no'], [1, 0])

# TODO: Set the number of training points 75%
num_train = 300

# Set the number of testing points 25%
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above

X_train, X_test, y_train ,y_test = train_test_split(X_all, y_all,test_size = num_test,train_size = num_train,random_state=1)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# TODO: Initialize the three models
clf_D = DecisionTreeClassifier(random_state=30)
clf_E = LogisticRegression(random_state=30)

clf_all = [clf_D, clf_E]

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]

train_set = [(X_train_100, y_train_100), (X_train_200, y_train_200), (X_train_300, y_train_300)]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

# TODO: Execute the 'train_predict' function for each classifier and each training set size

# train_predict(clf, X_train, y_train, X_test, y_test)
for clf in clf_all:
    for X, y in train_set:
        train_predict(clf, X, y, X_test, y_test)
        































