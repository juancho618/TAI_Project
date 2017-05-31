import pandas as pd
import matplotlib.pyplot as plt # plot library
from sklearn import tree



df = pd.read_csv('datacsvDays.csv', header=0)

features = list(df.columns[:6])
print features

y = df["Long Stay"]
X = df[features]
dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
