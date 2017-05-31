    from scipy.stats import spearmanr
    import pandas as pd

    df = pd.read_csv('datacsvDays.csv', header=0)
    print 'Age Correlation', spearmanr(df['Long Stay'], df['Age'])
    print 'Gender Correlation', spearmanr(df['Long Stay'], df['Gender'])
