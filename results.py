import os
import pandas as pd
import numpy as np
res_list = (os.listdir('data/models/'))

for res in res_list:
    weights = pd.read_csv('data/models/'+res)
    name = res.split('_')[0]
    alphas = pd.read_csv(f'data/company/{name}.csv')
    results = pd.DataFrame()
    print(weights.shape)
    print(alphas.shape)
    print(returns.shape)
    # for i in range(weights.shape[1]):
    #     print(weights.iloc[:,i])
    #     print(alphas.iloc[:,i+1])
    #     results[f'w_a_{i}'] = np.dot(alphas.iloc[:,i+1],weights.iloc[:,i])
    # print(results.head())
        