from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvHI.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
health_insurance = df['HI Classification'].value_counts().to_dict()
labels = ['Big', 'Medium', 'Small', 'Other']
items = [(v, k) for k, v in health_insurance.items()]
items.sort()
items.reverse()             # so largest is
print items
items = [(k, v) for v, k in items]
health_insurance_values = [x[1] for x in items]
print items
plt.bar(range(len(health_insurance)), health_insurance_values)
plt.xticks(range(len(labels)), labels)
plt.xlabel('Health Insurance Companies Classification')
plt.ylabel('Number of instances')
plt.title(r'Number of insances per Health Insurance Classify')
print 'Day of the week Correlation', spearmanr(df['Long Stay'], df['HI Classification'])
plt.show()
