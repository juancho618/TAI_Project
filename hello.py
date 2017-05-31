from sklearn import tree # importing tree classifier for sklearn


features = [[140, 1], [130, 1], [150, 0], [170, 0]] # intances with atribtes
labels = [0, 0, 1, 1]   # results of the traget variable
clf = tree.DecisionTreeClassifier() # calling the decision tree clasifier
clf = clf.fit(features, labels) # create the learninf instance
print clf.predict([[160,0]]), 'first classification: entries [160,0]' # send a paraeter to predict (classify)
print clf.predict([[145,0]]), 'first classification: entries [145,0]' # send a paraeter to predict (classify)
