from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
health_insurance = df['Health Insurance'].value_counts().to_dict()

items = [(v, k) for k, v in health_insurance.items()]
items.sort()
items.reverse()             # so largest is
print items
items = [(k, v) for v, k in items]
health_insurance_values = [x[1] for x in items]
print items
plt.bar(range(len(health_insurance)), health_insurance_values)
plt.xlabel('Health Insurance Companies')
plt.ylabel('Number of instances')
plt.title(r'Number of insances per Health Insurance')
plt.show()
