import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
age = df['Age']
prolonged = df['Age'].loc[df['Long Stay'] == True]
normal = df['Age'].loc[df['Long Stay'] == False]
plt.hist([prolonged, normal], stacked = True, color=['r','b'], label=['Prolonged', 'Expected'])
plt.xlabel('Age distribution - Prolonged and Expected Length of Stay')
plt.ylabel('Number of instances')
plt.title(r'Age distribution')
plt.legend(loc = "upper left")
plt.show()
print 'Correlation: ', df['Long Stay'].corr(df['Age'], method = 'pearson')
