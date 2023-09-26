# IMPORT ALL THE REQUIRED MODULE
from alphas import *
import alphas
import re
from helper_function import *
import pandas as pd
import numpy as np
import yfinance as yf



# DOWNLOAD ALL THE DATA 
print('\nImport Done')
start_date = '2013-01-01'
# end_date = '2023-09-25'
keys = ['TSLA' , 'GOOG' , 'META' , 'AAPL' , 'MSFT' , 'AMZN' , 'NVDA' , 'V' , 'ORCL' , 'CSCO']
open = pd.DataFrame(columns=keys)
high =pd.DataFrame(columns=keys)
low =pd.DataFrame(columns=keys)
close = pd.DataFrame(columns=keys)
volume = pd.DataFrame(columns=keys)
vwap = pd.DataFrame(columns=keys)
returns = pd.DataFrame(columns=keys)

print('\nLoading the data\n')
for s in keys:
    temp = yf.download(s,
                        start_date,
                        end_date=None)
    open[s] = temp['Open'].replace(np.nan ,0)
    high[s] = temp['High'].replace(np.nan ,0)
    low[s] = temp['Low'].replace(np.nan ,0)
    close[s] = temp['Close'].replace(np.nan ,0)
    volume[s] = temp['Volume'].replace(np.nan ,0)
    vwap[s] = ((temp['High'] + temp['Low'] + temp['Close'])/3).replace(np.nan ,0)
    returns[s] = temp.Close.pct_change().replace(np.nan ,0)

print(' Printing the Stats\n')
print('-'*20 + 'Data Shape' + '-'*20)

open.fillna(0 , inplace=True)
open.to_csv('data//base/open.csv', index=False)
print('open -' , open.shape)

high.fillna(0 , inplace=True)
high.to_csv('data/base/high.csv', index=False)
print('high -' , high.shape)

low.fillna(0 , inplace=True)
low.to_csv('data/base/low.csv', index=False)
print('low -' , low.shape)

close.fillna(0 , inplace=True)
print('close -' , close.shape)
close.to_csv('data/base/close.csv', index=False)

volume.fillna(0 , inplace=True)
print('volume -' , volume.shape)
volume.to_csv('data/base/volume.csv', index=False)

vwap.fillna(0 , inplace=True)
print('vwap -' , vwap.shape)
vwap.to_csv('data/base/vwap.csv', index=False)

returns.fillna(0 , inplace=True)
print('returns -' , returns.shape)
returns.to_csv('data/base/returns.csv', index=False)

print('-'*50)
print('\nApplying the functions\n')



# CREATING DATA FOR EACH COMPANY
def apply_functions_and_sum(o ,h ,l , c , vwap ,v , r, function_list, function_dict , comp):
    result_sum = pd.DataFrame()
    for i , func in enumerate(function_list):
        func_result = function_dict[func](o ,h ,l , c , vwap ,v , r)
        func_result.to_csv(f'data/alphas/{str(func)}.csv' , index=True)
        result_sum['A_'+str(i)] = func_result[comp]
    result_sum['close'] = c[comp]
    result_sum['open'] = o[comp]
    result_sum['high'] = h[comp]
    result_sum['low'] = l[comp]
    result_sum['volume'] = np.log(v[comp])
    result_sum['vwap'] = vwap[comp]
    result_sum['returns'] = r[comp]
    return result_sum

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

function_list = [i for i in dir(alphas) if 'alpha' in i]
function_list = sorted(function_list, key=natural_sort_key)
function_dict = {str(function_name): getattr(alphas, function_name) for function_name in function_list if callable(getattr(alphas, function_name))}
print('All the available alphas')
print(function_list)

print('\nCreating Company Data\n')
for col in list(open.columns):
    name = col
    temp = apply_functions_and_sum(open , high , low , close , vwap , volume , returns, function_list ,function_dict ,col)
    temp.dropna(inplace=True)
    temp.to_csv(f'data/company/{name}.csv',index=True)
    print(f'Company {name} created -> {temp.shape} ')
print('\nData Created\n')