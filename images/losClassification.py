import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
age = df.iloc[:,1]
los_Status =  df['Long Stay'].value_counts().to_dict()
x = ('Normal', 'Prolonged')
plt.bar(range(len(los_Status)),los_Status.values(),color=['b', 'r'])
plt.xticks(range(len(los_Status)), x)
plt.ylabel('Number of instances')
plt.xlabel('Length of stay classification')
plt.title(r'Length of stay classification distribution')
for i,item in enumerate(los_Status.values()):
    plt.text(i - .05, item + 50, str(item), fontdict=font)
plt.show()
