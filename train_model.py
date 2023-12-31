import model
from model import *
import re
import os


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


model_list = [i for i in dir(model) if ('mod' == i[:3] and (i[-1]) in [str(i) for i in range(10)])]
model_list = sorted(model_list, key=natural_sort_key)
model_dict = {str(function_name): getattr(model, function_name) for function_name in model_list if callable(getattr(model, function_name))}


def train_test(model , name, comp , X , y , test_size , frame_size ):
    train_size = int(frame_size*(1-test_size))
    val_df = pd.DataFrame(columns=y.columns)
    test_result = pd.DataFrame(columns=['weighted' , 'equal' , 'returns'])
    print()
    print('-'*30)
    print()
    print(f' STARTED TRAINING FOR {comp}')
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
        model.fit(X_train , y_train, epochs=30 , verbose=0)
        y_pred = pd.DataFrame(model.predict(tempx , verbose=0) , columns=y_test.columns)
        val_df = pd.concat([val_df , y_pred] , axis=0)

        y_pred1 = pd.DataFrame(model.predict(X_test, verbose=0) , columns=y_test.columns)
        cols = [i[2:] for i in y_test.columns]
        temp_equal = X_test[cols].sum(axis=1)
        temp_returns = X_test['returns'].shift(-1)
        temp_df = pd.DataFrame()
        for i in y_test.columns:
            temp_df[i[2:]] = y_pred1[i] * list(X_test[i[2:]])
        temp_weight = temp_df.sum(axis=1)
        df1 = pd.DataFrame({
                'weighted': list(temp_weight) ,
                'equal': list(temp_equal),
                'returns':list(temp_returns)
        })
        df1.dropna(inplace=True)
        test_result = test_result.append(df1,ignore_index=True)

    
    print(f' COMPLETED THE TRAINING FOR {comp} ')
    val_df.to_csv(f'./data/models/{comp}_{name}_result.csv',index=False)
    test_result.to_csv(f'./data/model_test/{comp}_{name}_result.csv',index=False)
    print(f' SAVED THE RESULTS TO ./data/models/{comp}_{name}_result.csv')
    print(f' SAVED THE RESULTS TO ./data/model_test/{comp}_{name}_result.csv')
    print()
    print('-'*30)


comp_list = [i.split('.')[0] for i in os.listdir('data/company')]


def softmax(df):
    temp_df = []
    for i in range(df.shape[0]):
        a = df.iloc[i,:]
        su = sum(a)
        x = []
        for i in a:
            if su != 0:
                x.append(i/su)
            else:
                x.append(0)
        temp_df.append(x)
    return pd.DataFrame(temp_df,columns=df.columns)


def create_data(df):
    lis = list(df.columns)
    # get all the alpha values ( known data )
    lis = [i for i in lis if 'A_' in i]
    
    for i in lis:
        df['y_'+i] = df[i]*(df['returns'].shift(-1))
        df['y_'+i] = df['y_'+i].apply(lambda x: x if x > 0 else 0)
        
    # target variable column
    lis = ['y_'+i for i in lis]
    df.dropna(inplace=True)
    return df.drop(lis , axis=1) , softmax(df[lis])

model_list = [i for i in dir(model) if ('mod' == i[:3] and (i[-1]) in [str(i) for i in range(10)])]
model_list = sorted(model_list, key=natural_sort_key)
model_dict = {str(function_name): getattr(model, function_name) for function_name in model_list if callable(getattr(model, function_name))}

for comp in comp_list:
    X , y = create_data(pd.read_csv('data/company/'+comp+'.csv'))
    print(comp)
    print('All the models - ')
    print(model_list)
    for model in model_list:
        print('*'*30)
        print(model)
        x = model_dict[model](X.shape[1]-1 , y.shape[1])
        train_test(x , model , comp , X , y , test_size=0.2 , frame_size=30 )
        print('*'*30)