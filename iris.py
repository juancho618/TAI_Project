import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print iris.feature_names
print iris.target_names

test_idx = [0,50,100]

# traning datasets
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

print train_data

# testting data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier() # calling the decision tree clasifier
clf.fit(train_data, train_target) # create the learninf instance

print test_target # test data result
print clf.predict(test_data) # grapich results

# Graph code
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                    out_file = dot_data,
                    feature_names = iris.feature_names,
                    class_names = iris.target_names,
                    filled = True, rounded = True,
                    impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
