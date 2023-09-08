import os
import pandas as pd

res_list = (os.listdir('data/models/'))

for res in res_list:
    temp = pd.read_csv('data/models/'+res)
    print(temp.columns)