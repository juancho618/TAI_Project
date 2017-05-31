import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('../losDist.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }
los = df['Length of Stay']

plt.boxplot([los], patch_artist = True)
plt.ylabel('Length of stay (Days)')
plt.title(r'Length of stay boxplot distribution')
plt.axvline(los.mean(), color='r', linestyle='dashed', linewidth=2)
plt.text(los.mean() + 0.5, 1400, str(math.ceil(los.mean())) + ' days', fontdict=font)
plt.show()
