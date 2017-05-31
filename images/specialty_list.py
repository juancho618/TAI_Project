from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

df = pd.read_csv('../datasetDiagnosisCSV.csv', header=0)
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 8,
        }

diagnosis_list = df['Speciality'].unique()
#writer = pd.ExcelWriter('list_diagnosis.xlsx', engine='xlsxwriter')
dtx = pd.DataFrame(diagnosis_list, columns = ["Speciality"])
#dtx.to_excel(writer, index = False)
#writer.save()
dtx.to_csv('specialty.csv', index = False)
