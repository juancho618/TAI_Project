import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'white',
        'weight': 'bold',
        'size': 12,
        }
age = df.iloc[:,1]
gender =  df['Gender'].value_counts().to_dict()
x = ('Male', 'Female')
print gender
plt.bar(range(len(gender)),gender.values(),color=['b', 'r'])
plt.xticks(range(len(gender)), x)
plt.ylabel('Number of instances')
plt.xlabel('Gender')
plt.title(r'Gender data balance')
for i,item in enumerate(gender.values()):
    plt.text(i - .05, 2000, str(item), fontdict=font)
plt.show()
