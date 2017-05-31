from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

df = pd.read_csv('../finalDataset.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
specialty = df['Speciality'].value_counts().to_dict()
specialty_n = df['Speciality'].unique()
print specialty
print 'size: ',len(specialty)
items = [(v, k) for k, v in specialty.items()]
items.sort()
items.reverse()             # so largest is
items = [(k, v) for v, k in items]
specialty_values = [x[1] for x in items]
plt.bar(range(len(specialty)), specialty_values)
plt.xlabel('Medical Specialty')
plt.ylabel('Number of instances')
plt.title(r'Number of instances per Medical Speciaty')
print 'Sopecialty Correlation', spearmanr(df['Long Stay'], df['Speciality'])
plt.show()
