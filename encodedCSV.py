import pandas as pd
import matplotlib.pyplot as plt # plot library
from sklearn import tree
from dataProcessing import * # dummy encoding
from sklearn.preprocessing import OneHotEncoder,LabelEncoder # best codification for categorical data


df = pd.read_csv('finalDataset.csv', header=0)
df2 = df.apply(LabelEncoder().fit_transform)
# print df # preprocess dataset with original values

# Convert all the nominal values to integers (dummy version)



#writer = pd.ExcelWriter('list_diagnosis.xlsx', engine='xlsxwriter')
dtx = pd.DataFrame(df2)
#dtx.to_excel(writer, index = False)
#writer.save()
dtx.to_csv('encodeCSV.csv', index = False)
