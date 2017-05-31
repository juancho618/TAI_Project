import pandas as pd
import matplotlib.pyplot as plt # plot library
import numpy as np
import itertools
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

# confusion matrix implementation
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from sklearn.metrics import confusion_matrix

c_matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

# Plot non-normalized confusion matrix
# plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c_matrix, classes=["Acceptable", "Prolonged"], normalize=False,
                      title='Confusion matrix Tree Classifier')

plt.show()

print(c_matrix)
