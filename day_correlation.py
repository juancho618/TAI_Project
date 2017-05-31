from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
day = df['Day']
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
prolonged_list=[]
expected_list = []
ind = np.arange(len(day_labels))
print ind

for i in day_labels:
    prolonged = df[(df['Long Stay'] == True) & (df['Day'] == i) ].count()
    expected = df[(df['Long Stay'] == False) & (df['Day'] == i) ].count()
    total = float(prolonged['Day'] + expected['Day'])
    prolonged_list.append(round(prolonged['Day']/total,2))
    expected_list.append(round(expected['Day']/total, 2))

print 'Prolonged: ', prolonged_list
print 'Expected: ',expected_list

p1 = plt.bar(ind, prolonged_list, color = ['b'], label = 'Prolonged')
p2 = plt.bar(ind, expected_list, color = ['r'], bottom = prolonged_list, label= 'Expected')
plt.xlabel('Day of the week')
plt.xticks(ind, day_labels)
plt.ylabel('Proportion of instances')
plt.title(r'Day of the week entry vs Length of Stay')
plt.legend(loc = "upper left")
print 'Day of the week Correlation', spearmanr(df['Long Stay'], df['Day'])
plt.show()
