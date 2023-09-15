import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

res_list = (os.listdir('data/models/'))

headings = ['Company' , 'Model' , 'Buy and Hold' , 'Equal weights' , 'Using DL Model']
values = []

T = 100

def calculate_returns(arr,  a):
    prod = a
    res = []
    for i in arr:
        res.append((1+i)*prod)
        prod *= (1+i)
    return prod , res

for res in res_list:    
    weights = pd.read_csv('data/models/'+res)
    name = res.split('_')[0]
    model_name = res.split('_')[1]
    alphas = pd.read_csv(f'data/company/{name}.csv')
    
    
    a_w = pd.DataFrame()
    for col in weights.columns:
        #            weights    *  alpha
        a_w[col] = weights[col] * alphas[col[2:]] 
    #  sum (weights*alpha)
    s_a_w = a_w.sum(axis=1)
    #  returns * sum (weights*alpha)
    r_s_a_w = (s_a_w*alphas['returns'].shift(-1))[:-1]
    r_s_a_w , r_s_a_w_arr = calculate_returns(r_s_a_w , T/10)
    
    
    alp = list(weights.columns)
    alp = [i[2:] for i in alp]
    #  sum( alphas )
    s_a = alphas[alp].sum(axis=1)
    # returns * sum( alphas )
    r_s_a = (s_a*alphas['returns'].shift(-1))[:-1]
    r_s_a , r_s_a_arr = calculate_returns(r_s_a , T/100)
    
    x = list(alphas['close'])
    b_h = (T/10)*(x[-1] - x[0])/x[0]
    
    values.append([name , model_name , b_h  , r_s_a , r_s_a_w])

    plt.title(name + ' ' + model_name)
    plt.plot(r_s_a_arr)
    plt.plot(r_s_a_w_arr)
    plt.legend(['equal weight' , 'Using DL Model'])    
    plt.show()

tab = PrettyTable(headings)
tab.add_rows(values)
print(tab)