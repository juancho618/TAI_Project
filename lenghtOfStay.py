import pandas as pd
import matplotlib.pyplot as plt # plot library
from sklearn import tree
from dataProcessing import * # dummy encoding
from sklearn.preprocessing import OneHotEncoder # best codification for categorical data

df = pd.read_csv('datacsvDays.csv', header=0)

# print df # preprocess dataset with original values

# Convert all the categorical values to integers
df2 = encodeColumnDummy(df,0) # changing health insurance
df2 = encodeColumnDummy(df2, 2) # changing Diagnosis
df2 = encodeColumnDummy(df2, 3) # changing Speciality
df2 = encodeColumnDummy(df2, 4) # changing Gender
df2 = encodeColumnDummy(df2, 5) # changing Day



# print df2

# -------- first way to get the data ------
# Option 1 (Manually)
# training data
train_data = df2.iloc[:5000,:6] # first 5000 intances as training data
train_target = df2.iloc[:5000, 6] # first 5000 target values


# testing data
test_data = df2.iloc[5001:,:6]
test_target = df2.iloc[5001:, 6]

# ------- functional way to get the data --------
x = df2.iloc[:,:6]  # data with the attributes
y = df2.iloc[:,6]   # results

# Option 2 (Build in Function)
# dataset spliter
from sklearn.cross_validation import train_test_split # import split and test functionality
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .6) # 60% of data for training

# ---- End functional ----

# tree classifier algorithm
clf = tree.DecisionTreeClassifier() # calling the decision tree clasifier
clf.fit(x_train, y_train) # create the learning instance

# second tree classifier
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)

# knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_classifier = knn.fit(x_train, y_train)

# bayes classifier
from sklearn.naive_bayes import GaussianNB # import gaussian classi
gnb = GaussianNB()
naive_bayes_classifier = gnb.fit(x_train, y_train)

# prediction 1
predictions = clf.predict(test_data)

# prediction 2
predictions2 = my_classifier.predict(x_test)

# naive bayes prediction
nb_predictions = naive_bayes_classifier.predict(test_data)

# naive bayes rnd data prediction
nb_predictions2 = naive_bayes_classifier.predict(x_test)
# naive bayes rnd data prediction
knn_predictions = knn_classifier.predict(x_test)

# to review accuracy
from sklearn.metrics import accuracy_score # impor accuracy score functionality
print 'accuracy tree fix data prediction',accuracy_score(test_target, predictions)
print 'accuracy tree random prediction',accuracy_score(y_test, predictions2)
print 'accuracy NB fix data prediction',accuracy_score(test_target, nb_predictions)
print 'accuracy NB random data prediction',accuracy_score(y_test, nb_predictions2)
print 'accuracy knn random data prediction',accuracy_score(y_test, knn_predictions)



'''
# Graph code
# features
features = list(df.columns[:6])

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                    out_file = dot_data,
                    feature_names = features,
                    class_names = ['Pronlonged', 'Acceptable'],
                    filled = True, rounded = True,
                    impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("LoS.pdf")
'''


# ------------------------- comments with useful lines ---------------------------

# Comparative histogram
#plt.hist([var1, var2], Stacked = True, color = ['r', 'b'])

# print df[1:3]
# print df.iloc[:7,:6] # print the first 7 rows taking into account 6 columns (attributes)


# print df.head() # print the first 4 elements

# print df.shape # print the shape of the data divided in (<total_instances>, <number_attributes>)

# print df.describe() # describes numerical data


# df.hist()
# plt.show()

# df['Age'].plot()
# plt.show()
