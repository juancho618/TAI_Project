from sklearn.preprocessing import OneHotEncoder # best codification for categorical data

def encodeColumnDummy(df, column):
    df_copy = df.copy()
    values = df_copy.iloc[:,column].unique()
    map_to_int = {name: n for n, name in enumerate(values)}
    df_copy.iloc[:,column].replace(map_to_int, inplace = True)

    return df_copy

def encodeColumnOneHot(df):
    df_dummy = df.copy()
    enc = OneHotEncoder()
    enc.fit(df.iloc[:,:])
    df_dummy = enc.n_values_
    '''
    df_dummy = encodeColumnDummy(df, col)
    enc = OneHotEncoder()
    enc.fit(df_dummy.iloc[:,col])
    df_dummy.iloc[:,col] = enc.n_values_
    '''
    return df_dummy
