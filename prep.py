######################################
# parameters
######################################
early_stopping_rounds=10
bin_act={'char_1':50,'char_2':50,'char_3':50,'char_4':50,'char_5':50,'char_6':50,'char_7':50,'char_8':50,'char_9':50,'char_10':100}

bin_peo={'group_1':800,'pchar_3':50,'pchar_4':50}
sample_frac=0.2
eta=0.1
num_round = 3000
tree_build_subsample=0.7
col_sample_tree=1

######################################
# functions
######################################
#do not take NA
def value_to_cat(series,num_chunk):
    if True in series.isnull().values:
        print('value_to_cat cannot handle NA')
        return []
    series_sorted=series.sort_values()
    boundary=[]
    for i in [ int(np.ceil(j*len(series)/num_chunk)-1) for j in range(num_chunk,0,-1)]:
        series_sorted.iloc[:i]=i
    series_sorted.sort_index(inplace=True)
    return series_sorted

def hist_bin(pds,bin_size):
    hist_b=pds.value_counts()
    hist=((hist_b.cumsum()-hist_b/2)/hist_b.sum()*bin_size).apply(np.ceil).astype('int32')
    print('{}{}'.format('Actual bin number: ',len(np.unique(hist.values))))
    return hist

def col_map(col,data,data_total,bin_info):
    bin_size=int(bin_info[col])
    red_ind=hist_bin(data_total[col],bin_size)
    return list(data[col].apply(lambda x: red_ind.loc[x]))
######################################
# read in data
######################################
act_data=pd.read_csv('act_train.csv',parse_dates=['date'])
peo_data=pd.read_csv('people.csv',parse_dates=['date'])
act_test=pd.read_csv('act_test.csv',parse_dates=['date'])
act_total=act_data.append(act_test)
data_temp=act_total.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
data_temp.replace(to_replace='.* ',regex=True,value='',inplace=True)
#############################################
# feature engineer
#############################################
gp=data_temp.groupby('group_1')
gp_first_register=gp['date_y'].min()
gp_first_register_year=gp_first_register.dt.year
gp_first_register_month=gp_first_register.dt.month
gp_first_act=gp['date_x'].min()
gp_first_act_year=gp_first_act.dt.year
gp_first_act_month=gp_first_act.dt.month
gp_last_act=gp['date_x'].max()
gp_last_act_year=gp_last_act.dt.year
gp_last_act_month=gp_last_act.dt.month
gp_num_act=gp['activity_id'].count()
gp_len_act=(gp['date_x'].max()-gp['date_x'].min()).dt.days+1
gp_freq_act=gp_num_act/gp_len_act

gp=peo_data.groupby('group_1')
gp_num_peo=gp['people_id'].count()
gp_is_small=gp_num_peo==1
# need to create product feature of gp_success_rt later

peo=data_temp.groupby('people_id')
peo_first_act=peo['date_x'].min()
peo_first_act_year=peo_first_act.dt.year
peo_first_act_month=peo_first_act.dt.month
peo_last_act=peo['date_x'].max()
peo_last_act_year=peo_last_act.dt.year
peo_last_act_month=peo_last_act.dt.month
peo_num_act=peo['activity_id'].count()
peo_one_act=peo_num_act==1
peo_len_act=(peo['date_x'].max()-peo['date_x'].min()).dt.days+1
peo_freq_act=peo_num_act/peo_len_act

ch10=data_temp.groupby('char_10_x')
ch10_success_rt=ch10['outcome'].mean()

gp_first_register.name='gp_first_register'
gp_first_register_year.name='gp_first_register_year'
gp_first_register_month.name='gp_first_register_month'
gp_first_act.name='gp_first_act'
gp_first_act_year.name='gp_first_act_year'
gp_first_act_month.name='gp_first_act_month'
gp_last_act.name='gp_last_act'
gp_last_act_year.name='gp_last_act_year'
gp_last_act_month.name='gp_last_act_month'
gp_num_act.name='gp_num_act'
gp_len_act.name='gp_len_act'
gp_freq_act.name='gp_freq_act'
gp_num_peo.name='gp_num_peo'
gp_is_small.name='gp_is_small'
peo_first_act.name='peo_first_act'
peo_first_act_year.name='peo_first_act_year'
peo_first_act_month.name='peo_first_act_month'
peo_last_act.name='peo_last_act'
peo_last_act_year.name='peo_last_act_year'
peo_last_act_month.name='peo_last_act_month'
peo_num_act.name='peo_num_act'
peo_one_act.name='peo_one_act'
peo_len_act.name='peo_len_act'
peo_freq_act.name='peo_freq_act'
ch10_success_rt.name='ch10_success_rt'

gp_feature=[gp_first_register,gp_first_register_year,gp_first_register_month,gp_first_act,gp_first_act_year,gp_first_act_month,gp_last_act,gp_last_act_year,gp_last_act_month,gp_num_act,gp_len_act,gp_freq_act,gp_num_peo,gp_is_small]
for feature in gp_feature:
    data_temp=data_temp.merge(pd.DataFrame(feature),how='left',left_on='group_1',right_index=True)
peo_feature=[peo_first_act,peo_first_act_year,peo_first_act_month,peo_last_act,peo_last_act_year,peo_last_act_month,peo_num_act,peo_one_act,peo_len_act,peo_freq_act]
for feature in peo_feature:
    data_temp=data_temp.merge(pd.DataFrame(feature),how='left',left_on='people_id',right_index=True)
data_temp=data_temp.merge(pd.DataFrame(ch10_success_rt),how='left',left_on='char_10_x',right_index=True)

data_temp['peo_len_vs_gp_len']=data_temp['peo_len_act']/data_temp['gp_len_act']
data_temp['peo_late_gp']=(data_temp['peo_first_act']-data_temp['gp_first_act']).dt.days
data_temp['gp_gap_act_register']=(data_temp['gp_first_act']-data_temp['gp_first_register']).dt.days
data_temp['peo_gap_act_register']=(data_temp['peo_first_act']-data_temp['date_y']).dt.days
data_temp['is_last_act']=data_temp['date_x']==data_temp['peo_last_act']
data_temp['act_year']=data_temp['date_x'].dt.year
data_temp['act_month']=data_temp['date_x'].dt.month
data_temp['act_day']=data_temp['date_x'].dt.day
data_temp['act_weekday']=data_temp['date_x'].dt.weekday
data_temp['peo_register_year']=data_temp['date_y'].dt.year
data_temp['peo_register_month']=data_temp['date_y'].dt.month
######################################
# product features
######################################
trdata_temp=act_data.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
gp=trdata_temp.groupby('group_1')
gp_success_rt=gp['outcome'].mean()
gp_success_rt.name='gp_success_rt'
data_temp=data_temp.merge(pd.DataFrame(gp_success_rt),how='left',left_on='group_1',right_index=True)
data_temp['is_success_gp']=data_temp['gp_success_rt']>=0.5
data_temp['is_unsuccess_gp']=data_temp['gp_success_rt']<0.5
######################################
del data_temp['people_id']
del data_temp['activity_id']
del data_temp['group_1']
data_temp.fillna(-1,inplace=True)
for col in data_temp.columns:
    if data_temp[col].dtype=='O':
        data_temp[col]=pd.to_numeric(data_temp[col],errors='coerce')
        data_temp[col]=data_temp[col].astype('int32',errors='ignore')

#we can first train those with gp_success_rt, and then train those without.
#we can further seperate those without into small groups and nonsmall groups.
######################################
# train test split
######################################
peo_test=peo_data.sample(frac=sample_frac)











######################################
#  multiprocessing by pool.map
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
