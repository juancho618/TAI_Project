import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
age = df['Age' & (df['Long Stay'] ==0)]

plt.hist([age])
plt.xlabel('Age (years)')
plt.ylabel('Number of instances')
plt.title(r'Age distribution')
plt.axvline(age.mean(), color='r', linestyle='dashed', linewidth=2)
plt.text(age.mean() + 0.5, 1400, str(round(age.mean())) + ' years', fontdict=font)
plt.show()
'''
los_Status =  df['Long Stay'].value_counts().to_dict()
x = ('Normal', 'Prolonged')
plt.bar(range(len(los_Status)),los_Status.values(),color=['b', 'r'])
plt.xticks(range(len(los_Status)), x)
plt.ylabel('Number of instances')
plt.xlabel('Lenght of stay classification')
plt.title(r'Lenght of stay classification distribution')
for i,item in enumerate(los_Status.values()):
    plt.text(i - .05, item + 50, str(item), fontdict=font)
plt.show()
'''

'''
# ------ Balance chart Lenght of stay -------


plt.show()
# plt.hist([age], stacked = True, color = ['b'])
# plt.show()
'''
