import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datacsvDays.csv', header=0)
font = {'family': 'serif',
        'color':  'white',
        'weight': 'bold',
        'size': 10,
        }
day = {}
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for d in day_labels:
    data = df[(df['Day']==d)].count()
    day.update({d:data['Day']})
x = day.keys()
print day
plt.bar(range(len(day)),day.values())
plt.xticks(range(len(day)), x)
plt.ylabel('Number of instances')
plt.xlabel('Day of the week')
plt.title(r'Day of the week patient entry distribution')
for i,item in enumerate(day.values()):
    plt.text(i - .2, 500, str(item), fontdict=font)
plt.show()
