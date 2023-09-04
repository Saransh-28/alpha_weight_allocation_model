import model
from model import *
import re
import os

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


model_list = [i for i in dir(model) if ('mod' == i[:3] and (i[-1]) in [str(i) for i in range(10)])]
model_list = sorted(model_list, key=natural_sort_key)
model_dict = {str(function_name): getattr(model, function_name) for function_name in model_list if callable(getattr(model, function_name))}


def train_test(model , name , X , y , test_size , frame_size):
    train_size = int(frame_size*(1-test_size))
    val_df = pd.DataFrame(columns=X.columns)
    print('\n\n\n')
    print('In the function')
    for i in range(0 , len(X) , frame_size):
        if (len(X) - i) <= frame_size:
            train_size = int((len(X) - i)*(1-test_size))
            tempx = X.iloc[i:len(X) , 1:]
            tempy = y[i:len(X)]
        else:
            tempx = X.iloc[i:i+frame_size , 1:]
            tempy = y[i:i+frame_size]
        X_train = tempx[:train_size]
        X_test = tempx[train_size:]
        y_train = tempy[:train_size]
        y_test = tempy[train_size:]
        X_train = X_train.astype('float32') 
        y_train = y_train.astype('float32') 
        X_test = X_test.astype('float32') 
        y_test = y_test.astype('float32') 
        print('training the model')
        model.fit(X_train , y_train, epochs=30 , verbose=1)
        y_pred = pd.DataFrame(model.predict(X_test) , columns=y_test.columns)
        print(f'epoch {i+1} -> {model.evaluate(X_test , y_test)}')
        val_df = pd.concat([val_df , y_pred] , axis=1)
    val_df.to_csv(f'./data/models/{name}_result.csv',index=False)

comp_list = [i.split('.')[0] for i in os.listdir('data/company')]

def create_data(df):
    lis = list(df.columns)
    lis = [i for i in lis if 'A_' in i]
    for i in lis:
        df['y_'+i] = df[i].shift(-1)
    lis = ['y_'+i for i in lis]
    df.dropna(inplace=True)
    return df.drop(lis , axis=1) , df[lis]

for comp in comp_list:
    model_list = [i for i in dir(model) if ('mod' == i[:3] and (i[-1]) in [str(i) for i in range(10)])]
    model_list = sorted(model_list, key=natural_sort_key)
    model_dict = {str(function_name): getattr(model, function_name) for function_name in model_list if callable(getattr(model, function_name))}
    X , y = create_data(pd.read_csv('data/company/'+comp+'.csv'))
    for model in model_list:
        print(model)
        x = model_dict[model](X.shape[1]-1 , y.shape[1])
        # print(x.summary())
        train_test(x , model , X , y , test_size=0.2 , frame_size=30)