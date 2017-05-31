import pandas as pd
import matplotlib.pyplot as plt # plot library
from sklearn import tree
from dataProcessing import * # dummy encoding
from sklearn.preprocessing import OneHotEncoder # best codification for categorical data

df = pd.read_csv('../finalDS.csv', header=0)

# print df # preprocess dataset with original values

# Convert all the nominal values to integers (dummy version)
df2 = encodeColumnDummy(df,0) # changing health insurance
df2 = encodeColumnDummy(df2, 2) # changing Diagnosis
df2 = encodeColumnDummy(df2, 3) # changing Speciality
df2 = encodeColumnDummy(df2, 4) # changing Gender
df2 = encodeColumnDummy(df2, 5) # changing Day

# ------- functional way to get the data --------
x = df2.iloc[:,:6]  # data with the attributes
y = df2.iloc[:,6]   # results

# dataset spliter
from sklearn.model_selection import train_test_split # import split and test functionality
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .8)


# tree classifier algorithm
clf = tree.DecisionTreeClassifier() # calling the decision tree clasifier

# Naive Bayes classifier algorithm
from sklearn.naive_bayes import MultinomialNB # import gaussian classi
nb_clf = MultinomialNB()


# --- Trying one hot encoder ------
enc = OneHotEncoder(categorical_features =[0, 2, 3, 4, 5]) # One Hot encoder Specifying the categorical attributes.
enc.fit(x) #fit the encoder to the data
clf.fit(enc.transform(x_train), y_train) # create the learninf instance
nb_clf.fit(enc.transform(x_train), y_train) # Nive Bayes - Multinomial model

# prediction
predictions = clf.predict(enc.transform(x_test))
prediction_NB = nb_clf.predict(enc.transform(x_test))

# import the result metrics
from sklearn import metrics

print("Tree Classifier Precision", metrics.precision_score(y_test, predictions))
print("Tree Classifier Recall", metrics.recall_score(y_test, predictions))
print("Tree Classifier Beta Score 1", metrics.f1_score(y_test, predictions))  
print("Tree Classifier Beta Score 0.5", metrics.fbeta_score(y_test, predictions, beta=0.5)) 
print("Tree Classifier Beta Score 2", metrics.fbeta_score(y_test, predictions, beta=2))

print("Naive Bayes Classifier Precision", metrics.precision_score(y_test,prediction_NB ))
print("Naive Bayes Recall", metrics.recall_score(y_test, prediction_NB))
print("Naive Bayes Beta Score 1", metrics.f1_score(y_test, prediction_NB))  
print("Naive Bayes Beta Score 0.5", metrics.fbeta_score(y_test, prediction_NB, beta=0.5)) 
print("Naive Bayes Beta Score 2", metrics.fbeta_score(y_test, prediction_NB, beta=2))  