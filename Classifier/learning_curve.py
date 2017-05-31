import pandas as pd
import matplotlib.pyplot as plt # plot library
from sklearn import tree
from dataProcessing import * # dummy encoding
from sklearn.preprocessing import OneHotEncoder # best codification for categorical data

df = pd.read_csv('../finalDataset.csv', header=0)

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
from sklearn.cross_validation import train_test_split # import split and test functionality
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5)


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


# Accuracy
from sklearn.metrics import accuracy_score # impor accuracy score functionality
print 'Accuracy tree encoded data prediction',accuracy_score(y_test, predictions)
print 'Accuracy Multinomial NB data prediction', accuracy_score(y_test, prediction_NB)

# Learning Curve plot
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



Xl, yl = enc.transform(x), y


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv1 = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = MultinomialNB()
plot_learning_curve(estimator, title, Xl, yl, ylim=(0.5, 1.01), cv=cv1, n_jobs=4)

title = "Learning Curves Decission Tree Classifier"
# SVC is more expensive so we do a lower number of CV iterations:
cv2 = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = tree.DecisionTreeClassifier()
plot_learning_curve(estimator, title, Xl, yl, (0.5, 1.01), cv=cv2, n_jobs=4)

plt.show()
