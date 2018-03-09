def act_process(data):
    data.replace(to_replace='.* ',regex=True,value='',inplace=True)
    data['month']=data['date'].dt.month
    data['day']=data['date'].dt.day
    data['weekday']=data['date'].dt.weekday
    data['date']=(data['date']-pd.datetime(2020,1,1))
    data['date']=data['date'].apply(lambda x: x.days)
    data.fillna(-1,inplace=True)
    for col in ['activity_category', 'char_1', 'char_2', 'char_3','char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10']:
        data[col]=pd.to_numeric(data[col],errors='coerce')
        data[col]=data[col].astype('int32',errors='ignore')

def hist_bin(pds,bin_size):
    hist_b=pds.value_counts()
    hist=((hist_b.cumsum()-hist_b/2)/hist_b.sum()*bin_size).apply(np.ceil).astype('int32')
    print('{}{}'.format('Actual bin number: ',len(np.unique(hist.values))))
    return hist

def col_map(col,data,data_total,bin_info):
    bin_size=int(bin_info[col])
    red_ind=hist_bin(data_total[col],bin_size)
    return list(data[col].apply(lambda x: red_ind.loc[x]))
###############################################
# parameters
###############################################
early_stopping_rounds=100
bin_act={'char_1':50,'char_2':50,'char_3':50,'char_4':50,'char_5':50,'char_6':50,'char_7':50,'char_8':50,'char_9':50,'char_10':100}


bin_peo={'group_1':800,'pchar_3':50,'pchar_4':50}
sample_frac=0.2
eta=0.1
num_round = 3000
tree_build_subsample=0.7
col_sample_tree=1

###############################################

act_data=pd.read_csv('act_train.csv',parse_dates=['date'])
peo_data=pd.read_csv('people.csv',parse_dates=['date'])
act_test=pd.read_csv('act_test.csv',parse_dates=['date'])

act_process(act_data)
act_process(act_test)

peo_data.columns=['people_id', 'pchar_1', 'group_1', 'pchar_2', 'date', 'pchar_3', 'pchar_4','pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9', 'pchar_10', 'pchar_11','pchar_12', 'pchar_13', 'pchar_14', 'pchar_15', 'pchar_16', 'pchar_17','pchar_18', 'pchar_19', 'pchar_20', 'pchar_21', 'pchar_22', 'pchar_23','pchar_24', 'pchar_25', 'pchar_26', 'pchar_27', 'pchar_28', 'pchar_29','pchar_30', 'pchar_31', 'pchar_32', 'pchar_33', 'pchar_34', 'pchar_35','pchar_36', 'pchar_37', 'pchar_38']
peo_data.replace(to_replace='.* ',regex=True,value='',inplace=True)
peo_data['date']=(peo_data['date']-pd.datetime(2020,1,1))
peo_data['date']=peo_data['date'].apply(lambda x: x.days)
for col in ['pchar_1', 'group_1', 'pchar_2', 'date', 'pchar_3','pchar_4', 'pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9']:
    peo_data[col]=pd.to_numeric(peo_data[col],errors='coerce')
    peo_data[col]=peo_data[col].astype('int32',errors='ignore')
for col in ['pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15','pchar_16', 'pchar_17','pchar_18', 'pchar_19', 'pchar_20', 'pchar_21','pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27','pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33','pchar_34', 'pchar_35', 'pchar_36', 'pchar_37']:
    peo_data[col]=peo_data[col].astype('int8')


######################################
#  multiprocessing by pool.starmap
######################################
red_act=list(bin_act.keys())
red_peo=list(bin_peo.keys())
act_total=act_data[red_act].append(act_test[red_act])
from functools import partial
with Pool(5) as p:
    red_temp=p.map(partial(col_map,data=act_data,data_total=act_total,bin_info=bin_act),red_act)
for i in range(len(red_act)):
    try:
        act_data.drop(columns=red_act[i]+'_red',inplace=True)
    except:
        print('no '+red_act[i]+'_red')
    act_data[red_act[i]+'_red']=red_temp[i]

with Pool(5) as p:
    red_temp=p.map(partial(col_map,data=act_test,data_total=act_total,bin_info=bin_act),red_act)
for i in range(len(red_act)):
    try:
        act_test.drop(columns=red_act[i]+'_red',inplace=True)
    except:
        print('no '+red_act[i]+'_red')
    act_test[red_act[i]+'_red']=red_temp[i]

with Pool(5) as p:
    red_temp=p.map(partial(col_map,data=peo_data,data_total=peo_data,bin_info=bin_peo),red_peo)
for i in range(len(red_peo)):
    try:
        peo_data.drop(columns=red_peo[i]+'_red',inplace=True)
    except:
        print('no '+red_peo[i]+'_red')
    peo_data[red_peo[i]+'_red']=red_temp[i]
######################################

use_act=act_data[['people_id', 'date', 'activity_category','outcome', 'month','day', 'weekday', 'char_1_red', 'char_2_red','char_3_red', 'char_4_red', 'char_5_red', 'char_6_red', 'char_7_red','char_8_red', 'char_9_red', 'char_10_red']]
use_test=act_test[['people_id', 'date', 'activity_category','month', 'day', 'weekday', 'char_1_red', 'char_2_red','char_3_red', 'char_4_red', 'char_5_red', 'char_6_red', 'char_7_red','char_8_red', 'char_9_red', 'char_10_red']]
use_peo=peo_data[['people_id', 'pchar_1','pchar_2', 'date','pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9',
'pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15','pchar_16', 'pchar_17', 'pchar_18', 'pchar_19', 'pchar_20', 'pchar_21','pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27',
'pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33','pchar_34', 'pchar_35', 'pchar_36', 'pchar_37', 'pchar_38','group_1_red', 'pchar_3_red', 'pchar_4_red']]

###########################################
# data comparison for reducing bias
###########################################
import os
os.system('rm -f use*')
for col in use_test.columns:
    (use_test[col].value_counts()/len(use_test)*100).to_csv('use_test.csv',mode='a',header=col)
for col in use_act.columns:
    (use_act[col].value_counts()/len(use_act)*100).to_csv('use_act.csv',mode='a',header=col)
for col in use_peo.columns:
    (use_peo[col].value_counts()/len(use_peo)*100).to_csv('use_peo.csv',mode='a',header=col)

#By compareing the training data and the testing data, we found that char_10_red 23 is over represented in training data. At the same time, the top five users in training data generated the majority of those instances. That is the reason why during our first training of model, the testing error rate using part of the training data is considerably higher than the error rate on Kaggle. We therefore remove those five users from the training data to reduce bias.

use_act=use_act[(use_act['people_id']!='ppl_294918') & (use_act['people_id']!='ppl_370270') & (use_act['people_id']!='ppl_105739') & (use_act['people_id']!='ppl_54699') & (use_act['people_id']!='ppl_64887') ]
##########################################
df=pd.merge(use_act,use_peo,how='left',left_on='people_id',right_on='people_id')
test_df=pd.merge(use_test,use_peo,how='left',left_on='people_id',right_on='people_id')

not_categorical=['date_x', 'date_y', 'pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15', 'pchar_16', 'pchar_17', 'pchar_18', 'pchar_19', 'pchar_20', 'pchar_21', 'pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27', 'pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33', 'pchar_34', 'pchar_35', 'pchar_36', 'pchar_37', 'pchar_38','outcome','people_id']

categorical=[]
for category in df.columns:
    if category not in not_categorical:
        categorical.append(category)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc=enc.fit(pd.concat([df[categorical],test_df[categorical]]))

###########################################
# sampling and training
###########################################
#Even after removing the top 5 users, the error rate is still far away from Kaggle error rate. That means we should sample on people_id. As it turns out, this is a much better indicator of real auc. This mean people_id are important.

#An efficient approach to train and test is to separate a small group of peoples as test. Among the remaining people, we can mix their activities. This will be a good mimic of error rate, and will improve our auc.

peo_test=peo_data.sample(frac=0.1)
df_test=df[df['people_id'].isin(peo_test['people_id'])]
res_test=df_test['outcome']

df_train=df[~df['people_id'].isin(peo_test['people_id'])]
df_train=df_train.sample(frac=sample_frac)
res_train=df_train['outcome']

df_cat_train=enc.transform(df_train[categorical])
df_cat_test=enc.transform(df_test[categorical])
test_df_cat=enc.transform(test_df[categorical])

from scipy.sparse import hstack
df_spr_train=hstack((df_train[not_categorical[:-2]],df_cat_train))
df_spr_test=hstack((df_test[not_categorical[:-2]],df_cat_test))
test_df_spr=hstack((test_df[not_categorical[:-2]],test_df_cat))
dtrain=xgb.DMatrix(df_spr_train,label=res_train)
dtest=xgb.DMatrix(df_spr_test)
test_dtest=xgb.DMatrix(test_df_spr)

watchlist  = [(dtrain,'train')]
from sklearn.metrics import roc_auc_score

param = {'max_depth':10, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
print(roc_auc_score(res_test,bst.predict(dtest)))

test_pred=bst.predict(test_dtest)
act_test2=pd.read_csv('act_test.csv',parse_dates=['date'])
output = pd.DataFrame({ 'activity_id' : act_test2['activity_id'], 'outcome': test_pred })
output.to_csv('redhat_pred.csv', index = False)
###########################################
# parameter tuning
###########################################
perf=[]
for num_round in [6000,10000,20000]:
    bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
    print(roc_auc_score(res_test,bst.predict(dtest)))
    perf.append(roc_auc_score(res_test,bst.predict(dtest)))
    
perf_final=pd.DataFrame(perf,index=[6000,10000,20000])
perf_final.to_csv('night_perf.csv', index = True)
