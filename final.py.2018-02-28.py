######################################
print('parameters')
######################################
char_10_x_bin=1000
sample_frac=0.5
early_stopping_rounds=20
eta=0.1
num_round_gp = 1000
num_round_no_gp=100
tree_build_subsample=1
col_sample_tree=1
######################################
print('functions')
######################################

def simple_data_treat(data):
    data.replace(to_replace='.* ',regex=True,value='',inplace=True)
    for col in data.columns:
        if (data[col].dtype=='O') & (col!='people_id') & (col!='group_1') & (col!='activity_id'):
            data[col]=pd.to_numeric(data[col],errors='coerce')
            data[col]=data[col].astype('int32',errors='ignore')
    for col in data.columns:
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    data.fillna(60000,inplace=True)#fill number cannot be negative, cannot be too large
    return data

def hist_bin(pds,bin_size):
    hist_b=pds.value_counts()
    hist=((hist_b.cumsum()-hist_b/2)/hist_b.sum()*bin_size).apply(np.ceil).astype('int32')
    print('{}{}'.format('Actual bin number: ',len(np.unique(hist.values))))
    return hist
######################################
print('read in data')
######################################
act_train=pd.read_csv('act_train.csv',parse_dates=['date'])
act_test=pd.read_csv('act_test.csv',parse_dates=['date'])
peo_data=pd.read_csv('people.csv',parse_dates=['date'])
#By compareing the training data and the testing data, we found that char_10_red 23 is over represented in training data. At the same time, the top five users in training data generated the majority of those instances. That is the reason why during our first training of model, the testing error rate using part of the training data is considerably higher than the error rate on Kaggle. We therefore remove those five users from the training data to reduce bias.
act_train=act_train[(act_train['people_id']!='ppl_294918') & (act_train['people_id']!='ppl_370270') & (act_train['people_id']!='ppl_105739') & (act_train['people_id']!='ppl_54699') & (act_train['people_id']!='ppl_64887') ]
act_total=act_train.append(act_test)
data_temp=act_total.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
#############################################
print('feature engineer')
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
print('product features')
######################################
trdata_temp=act_train.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
trdata_temp['group_1'].replace(to_replace='.* ',regex=True,value='',inplace=True)
gp=trdata_temp.groupby('group_1')
gp_success_rt=gp['outcome'].mean()
gp_success_rt.name='gp_success_rt'
data_temp=data_temp.merge(pd.DataFrame(gp_success_rt),how='left',left_on='group_1',right_index=True)
data_temp['is_success_gp']=data_temp['gp_success_rt']>=0.5
data_temp['is_unsuccess_gp']=data_temp['gp_success_rt']<0.5
data_temp['is_success_char_10']=data_temp['ch10_success_rt']>=0.5
data_temp['is_unsuccess_char_10']=data_temp['ch10_success_rt']<0.5
del trdata_temp
######################################
print('data observation: which categorical data should be reduced')
######################################
for col in ['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x',  'char_1_y','char_2_y',  'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_38']:    
    print(data_temp[col].value_counts()/len(data_temp)*100)
#apart from group_1, we probaby just need to reduce the size of char_10
char_10_x_red=hist_bin(data_temp['char_10_x'],char_10_x_bin)
char_10_x_red.name='char_10_x_red'
data_temp=data_temp.merge(pd.DataFrame(char_10_x_red),how='left',left_on='char_10_x',right_index=True)

#we can first train those with gp_success_rt, and then train those without.
#we can further seperate those without into small groups and nonsmall groups.
######################################
print('final data treatment')
######################################
peo_data=simple_data_treat(peo_data)
data_temp=simple_data_treat(data_temp)
data_train=data_temp.iloc[:len(act_train)]
data_test=data_temp.iloc[len(act_train):]
peo_train=peo_data[peo_data['people_id'].isin(data_train['people_id'])]
del data_temp
######################################
print('train test split')
######################################
#Even after removing the top 5 users, the error rate is still far away from Kaggle error rate. That means we should sample on people_id. As it turns out, this is a much better indicator of real auc. This mean people_id are important.
#An efficient approach to train and test is to separate a small group of peoples as test. Among the remaining people, we can mix their activities. This will be a good mimic of error rate, and will improve our auc.
peo_train_fraction=peo_train.sample(frac=sample_frac)
t_peo_train,t_peo_test=train_test_split(peo_train_fraction,test_size=0.2)
t_gp_intersect=np.intersect1d(t_peo_train['group_1'].values,t_peo_test['group_1'].values)
t_train_gp=data_train[data_train['people_id'].isin(t_peo_train['people_id']) & data_train['group_1'].isin(t_gp_intersect)]
t_train_no_gp=data_train[data_train['people_id'].isin(t_peo_train['people_id']) & ~(data_train['group_1'].isin(t_gp_intersect))]
t_test_gp=data_train[data_train['people_id'].isin(t_peo_test['people_id']) & data_train['group_1'].isin(t_gp_intersect)]
t_test_no_gp=data_train[data_train['people_id'].isin(t_peo_test['people_id']) & ~(data_train['group_1'].isin(t_gp_intersect))]
######################################
print('create df for training/testing')
######################################
available_col=['act_day','act_month','act_weekday','act_year','activity_category','activity_id','ch10_success_rt','char_10_x','char_10_x_red','char_10_y','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_1_x','char_1_y','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_2_x','char_2_y','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37','char_38','char_3_x','char_3_y','char_4_x','char_4_y','char_5_x','char_5_y','char_6_x','char_6_y','char_7_x','char_7_y','char_8_x','char_8_y','char_9_x','char_9_y','date_x','date_y','gp_first_act','gp_first_act_month','gp_first_act_year','gp_first_register','gp_first_register_month','gp_first_register_year','gp_freq_act','gp_gap_act_register','gp_is_small','gp_last_act','gp_last_act_month','gp_last_act_year','gp_len_act','gp_num_act','gp_num_peo','gp_success_rt','group_1','is_last_act','is_success_char_10','is_success_gp','is_unsuccess_char_10','is_unsuccess_gp','outcome','peo_first_act','peo_first_act_month','peo_first_act_year','peo_freq_act','peo_gap_act_register','peo_last_act','peo_last_act_month','peo_last_act_year','peo_late_gp','peo_len_act','peo_len_vs_gp_len','peo_num_act','peo_one_act','peo_register_month','peo_register_year', 'people_id']
categorical_col=['act_day','act_month','act_weekday','act_year','activity_category','char_10_x','char_10_x_red','char_1_x','char_2_x','char_2_y','char_3_x','char_3_y','char_4_x','char_4_y','char_5_x','char_5_y','char_6_x','char_6_y','char_7_x','char_7_y','char_8_x','char_8_y','char_9_x','char_9_y','gp_first_act_month','gp_first_act_year','gp_first_register_month','gp_first_register_year','gp_last_act_month','gp_last_act_year','group_1','peo_first_act_month','peo_first_act_year','peo_last_act_month','peo_last_act_year','peo_register_month','peo_register_year']

gp_use_col=['act_day','act_month','act_weekday','act_year','activity_category','ch10_success_rt','char_10_x','char_10_y','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_1_x','char_1_y','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_2_x','char_2_y','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37','char_38','char_3_x','char_3_y','char_4_x','char_4_y','char_5_x','char_5_y','char_6_x','char_6_y','char_7_x','char_7_y','char_8_x','char_8_y','char_9_x','char_9_y','gp_success_rt','group_1','is_last_act','is_success_char_10','is_success_gp','is_unsuccess_char_10','is_unsuccess_gp','peo_first_act_month','peo_first_act_year','peo_freq_act','peo_gap_act_register','peo_last_act_month','peo_last_act_year','peo_late_gp','peo_len_act','peo_len_vs_gp_len','peo_num_act','peo_one_act','peo_register_month','peo_register_year']
no_gp_use_col=['act_day','act_month','act_weekday','act_year','activity_category','ch10_success_rt','char_10_x','char_10_y','char_11','char_12','char_13','char_14','char_15','char_16','char_17','char_18','char_19','char_1_x','char_1_y','char_20','char_21','char_22','char_23','char_24','char_25','char_26','char_27','char_28','char_29','char_2_x','char_2_y','char_30','char_31','char_32','char_33','char_34','char_35','char_36','char_37','char_38','char_3_x','char_3_y','char_4_x','char_4_y','char_5_x','char_5_y','char_6_x','char_6_y','char_7_x','char_7_y','char_8_x','char_8_y','char_9_x','char_9_y','gp_first_act_month','gp_first_act_year','gp_first_register_month','gp_first_register_year','gp_freq_act','gp_gap_act_register','gp_is_small','gp_last_act_month','gp_last_act_year','gp_len_act','gp_num_act','gp_num_peo','is_last_act','is_success_char_10','is_unsuccess_char_10','peo_first_act_month','peo_first_act_year','peo_freq_act','peo_gap_act_register','peo_last_act_month','peo_last_act_year','peo_late_gp','peo_len_act','peo_len_vs_gp_len','peo_num_act','peo_one_act','peo_register_month','peo_register_year']

gp_not_use_col=list(set(t_train_gp.columns)-set(gp_use_col))
no_gp_not_use_col=list(set(t_train_gp.columns)-set(no_gp_use_col))

t_df_train_gp=t_train_gp[gp_use_col]
t_df_test_gp=t_test_gp[gp_use_col]
t_df_train_no_gp=t_train_no_gp[no_gp_use_col]
t_df_test_no_gp=t_test_no_gp[no_gp_use_col]
######################################
print('create OneHot')
######################################
categorical_gp=list(set(gp_use_col)&set(categorical_col))
categorical_no_gp=list(set(no_gp_use_col)&set(categorical_col))

not_categorical_gp=list(set(gp_use_col)-set(categorical_gp))
not_categorical_no_gp=list(set(no_gp_use_col)-set(categorical_no_gp))

from sklearn.preprocessing import OneHotEncoder
enc_gp=OneHotEncoder(handle_unknown='ignore')
enc_gp=enc_gp.fit(pd.concat([t_train_gp[categorical_gp],data_test[categorical_gp]]))
enc_no_gp=OneHotEncoder(handle_unknown='ignore')
enc_no_gp=enc_no_gp.fit(pd.concat([t_train_no_gp[categorical_no_gp],data_test[categorical_no_gp]]))
t_df_cat_train_gp=enc_gp.transform(t_df_train_gp[categorical_gp])
t_df_cat_test_gp=enc_gp.transform(t_df_test_gp[categorical_gp])
t_df_cat_train_no_gp=enc_no_gp.transform(t_df_train_no_gp[categorical_no_gp])
t_df_cat_test_no_gp=enc_no_gp.transform(t_df_test_no_gp[categorical_no_gp])

from scipy.sparse import hstack
t_df_spr_train_gp=hstack((t_df_train_gp[not_categorical_gp],t_df_cat_train_gp))
t_df_spr_test_gp=hstack((t_df_test_gp[not_categorical_gp],t_df_cat_test_gp))
t_df_spr_train_no_gp=hstack((t_df_train_no_gp[not_categorical_no_gp],t_df_cat_train_no_gp))
t_df_spr_test_no_gp=hstack((t_df_test_no_gp[not_categorical_no_gp],t_df_cat_test_no_gp))
t_dtrain_gp=xgb.DMatrix(t_df_spr_train_gp,label=t_train_gp['outcome'])
t_dtest_gp=xgb.DMatrix(t_df_spr_test_gp)
t_dtrain_no_gp=xgb.DMatrix(t_df_spr_train_no_gp,label=t_train_no_gp['outcome'])
t_dtest_no_gp=xgb.DMatrix(t_df_spr_test_no_gp)
######################################
print('run test')
######################################
from sklearn.metrics import roc_auc_score

watchlist_gp  = [(t_dtrain_gp,'train')]
param_gp = {'max_depth':12, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst_gp = xgb.train(param_gp, t_dtrain_gp, num_round_gp, watchlist_gp,early_stopping_rounds=early_stopping_rounds)
print(roc_auc_score(t_test_gp['outcome'],bst_gp.predict(t_dtest_gp)))

param_no_gp = {'max_depth':8, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
watchlist_no_gp  = [(t_dtrain_no_gp,'train')]
bst_no_gp = xgb.train(param_no_gp, t_dtrain_no_gp, num_round_no_gp, watchlist_no_gp,early_stopping_rounds=early_stopping_rounds)
print(roc_auc_score(t_test_no_gp['outcome'],bst_no_gp.predict(t_dtest_no_gp)))

print(roc_auc_score(np.concatenate((t_test_gp['outcome'],t_test_no_gp['outcome'])),np.concatenate((bst_gp.predict(t_dtest_gp),bst_no_gp.predict(t_dtest_no_gp)))))

######################################
print('output for Kaggle')
######################################
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

