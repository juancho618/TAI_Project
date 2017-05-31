from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

df = pd.read_csv('../datasetDiagnosisCSV.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
diagnosis = df['Diagnosis'].value_counts().to_dict()
diagnosis_list = df['Diagnosis'].unique()

print diagnosis_list
'''write the csv list
diagnosis_list = df['Diagnosis'].unique()
#writer = pd.ExcelWriter('list_diagnosis.xlsx', engine='xlsxwriter')
dtx = pd.DataFrame(diagnosis_list, columns = ["Diagnosis"])
#dtx.to_excel(writer, index = False)
#writer.save()
dtx.to_csv('hi.csv', index = False)
'''
items = [(v, k) for k, v in diagnosis.items()]
items.sort()
items.reverse()             # so largest is
items = [(k, v) for v, k in items]
diagnosis_values = [x[1] for x in items]
plt.bar(range(len(diagnosis)), diagnosis_values)
plt.xlabel('Diagnosis')
plt.ylabel('Number of instances')
plt.title(r'Number of instances per Diagnosis')
print 'Diagnosis Correlation', spearmanr(df['Long Stay'], df['Diagnosis'])
plt.show()
