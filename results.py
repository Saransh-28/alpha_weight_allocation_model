import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def res():
    res_list = (os.listdir('data/models/'))
    res_list.sort()

    headings = ['Company' , 'Model' , 'Buy and Hold' , 'Equal weights' , 'Equal Weights Win Rate' , 'Using DL Model', 'DL Model Win Rate']
    values = []

    T = 100

    def calculate_returns(arr,  a):
        prod = 1
        res = []
        for i in arr:
            res.append((1+i)*prod)
            prod *= (1+i)
        return a*prod , res
    def calculate_win_rate(arr):
        return len([i for i in arr if i>0])/len(arr)
    
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
        r_s_a_w = (s_a_w*(alphas['returns'].shift(-1)))[:-1]
        r_s_a_w_r=calculate_win_rate(r_s_a_w)
        r_s_a_w , r_s_a_w_arr = calculate_returns(r_s_a_w , T/10)
        
        
        alp = list(weights.columns)
        alp = [i[2:] for i in alp]
        #  sum( alphas )
        s_a = alphas[alp].sum(axis=1)
        # returns * sum( alphas )
        r_s_a = (s_a*alphas['returns'].shift(-1))[:-1]
        r_s_a_r=calculate_win_rate(r_s_a)
        r_s_a , r_s_a_arr = calculate_returns(r_s_a , T/100)
        
        x = list(alphas['close'])
        b_h = (T/10)*(x[-1] - x[0])/x[0]
        
        values.append([name , model_name , b_h  , r_s_a, r_s_a_r , r_s_a_w, r_s_a_w_r])

        # plt.title(name + ' ' + model_name)
        # plt.plot(r_s_a_arr)
        # plt.plot(r_s_a_w_arr)
        # plt.legend(['equal weight' , 'Using DL Model'])    
        # plt.show()

    print('+'+'-'*28 + '-'*29 +'-'*27+'+')
    print('|'+' '*28 + 'Final Result on whole dataset' +' '*27 + '|')
    tab = PrettyTable(headings)
    tab.add_rows(values)
    print(tab)

    headings = ['Company' , 'Model' , 'Equal weights (testing period) ' ,'Equal Weights Win Rate' , 'Using DL Model (testing period)', 'DL Model Win Rate']
    values = []

    res_list1 = (os.listdir('data/model_test/'))
    res_list1.sort()
    for res in res_list1: 
        df = pd.read_csv('data/model_test/'+res)
        name = res.split('_')[0]
        model = res.split('_')[1]
        df['weighted'] = df['weighted']*df['returns']
        df['equal'] = df['equal']*df['returns']
        weight_returns , weight_arr = calculate_returns(df['weighted'] , T/10)
        weight_win_rate=calculate_win_rate(df['weighted'])
        equal_returns , equal_arr = calculate_returns(df['equal'] , T/100)
        equal_win_rate=calculate_win_rate(df['equal'])
        values.append([name, model , equal_returns, equal_win_rate , weight_returns, weight_win_rate])


    print('+-'+'-'*28 + '-'*29 +'-'*27+'-+')
    print('| '+' '*28 + '   Final Result on test set  ' +' '*27 + ' |')
    tab = PrettyTable(headings)
    tab.add_rows(values)
    print(tab)


res()