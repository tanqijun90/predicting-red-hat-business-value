######################################
print('parameters')
######################################
sample_frac=0.5
early_stopping_rounds=10
eta=0.1
num_round_gp = 100
num_round_no_gp= 40
tree_build_subsample=1
col_sample_tree=1
######################################
print('functions')
######################################
def prod_feature(data,col1,col2):
    gpb=data.groupby(by=[col1,col2])
    new_index=gpb[col1].count()
    list_index=list(range(len(new_index)))
    new_index[:]=list_index
    new_index.name=col1+'_prod_'+col2
    new_index.astype('int32')
    data=data.merge(pd.DataFrame(new_index),how='left',left_on=[col1,col2],right_index=True)
    return data

def simple_data_treat(data):
    data.replace(to_replace='.* ',regex=True,value='',inplace=True)
    for col in data.columns:
        if (data[col].dtype=='O') & (col!='people_id') & (col!='group_1') & (col!='activity_id'):
            data[col]=pd.to_numeric(data[col],errors='coerce')
            data[col]=data[col].astype('int32',errors='ignore')
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    data.fillna(int(60000),inplace=True)#fill number cannot be negative, cannot be too large
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
m_total=act_total.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
m_train=m_total.iloc[:len(act_train)]
gp=m_train.groupby('group_1')
gp_num_act=gp['activity_id'].count()
gp_num_act.name='gp_num_act'
m_train=m_train.merge(pd.DataFrame(gp_num_act),how='left',left_on='group_1',right_index=True)
small_data=m_train[m_train['gp_num_act']<5000]
#############################################
print('feature engineer')
#############################################
#We need to engineer 
#char_10_x char_38 success rate
#char_3_y char_7_y product
#char_3_y_prod_char_7_y char_2_y success rate
#char_3_y_prod_char_7_y char_9_y success rate
#act_week_num
#act_week_num group_1 success rate
#group_1 success rate
#act_weekday,act_year,act_month,act_day
#is_last_act
#convert date_x,date_y into integers
#We also need to unmask some of the engineered features
#We also need to pay attension to the success rate for t_test, since some of them will not be available.
#gpb=small_data.groupby(['char_10_x','char_38'])
gpb=m_train.groupby(['char_10_x','char_38'])
char_10_x_char_38_rt=gpb['outcome'].mean()
char_10_x_char_38_rt.name='char_10_x_char_38_rt'
m_total=m_total.merge(pd.DataFrame(char_10_x_char_38_rt),how='left',left_on=['char_10_x','char_38'],right_index=True)

m_total=prod_feature(m_total,'char_3_y','char_7_y')
m_train=m_total.iloc[:len(act_train)]
gpb=m_train.groupby(['char_3_y_prod_char_7_y','char_2_y'])
char_3_7_2_rt=gpb['outcome'].mean()
char_3_7_2_rt.name='char_3_7_2_rt'
m_total=m_total.merge(pd.DataFrame(char_3_7_2_rt),how='left',left_on=['char_3_y_prod_char_7_y','char_2_y'],right_index=True)

gpb=m_train.groupby(['char_3_y_prod_char_7_y','char_9_y'])
char_3_7_9_rt=gpb['outcome'].mean()
char_3_7_9_rt.name='char_3_7_9_rt'
m_total=m_total.merge(pd.DataFrame(char_3_7_9_rt),how='left',left_on=['char_3_y_prod_char_7_y','char_9_y'],right_index=True)

m_total['act_week_num']=np.floor(((m_total['date_x']-pd.datetime(2020,1,7)).dt.days)/7)
m_total['act_week_num']=m_total['act_week_num'].astype('int32')
m_train=m_total.iloc[:len(act_train)]
gpb=m_train.groupby(['act_week_num','group_1'])
act_week_num_group_rt=gpb['outcome'].mean()
act_week_num_group_rt.name='act_week_num_group_rt'
m_total=m_total.merge(pd.DataFrame(act_week_num_group_rt),how='left',left_on=['act_week_num','group_1'],right_index=True)
#This is not a feature, but it will help us to train.
m_total=prod_feature(m_total,'act_week_num','group_1')

gpb=m_train.groupby(['group_1'])
group_rt=gpb['outcome'].mean()
group_rt.name='group_rt'
m_total=m_total.merge(pd.DataFrame(group_rt),how='left',left_on=['group_1'],right_index=True)

peo=m_total.groupby('people_id')
peo_last_act=peo['date_x'].max()
peo_last_act.name='peo_last_act'
m_total=m_total.merge(pd.DataFrame(peo_last_act),how='left',left_on=['people_id'],right_index=True)

m_total['is_last_act']=m_total['date_x']==m_total['peo_last_act']
m_total['act_year']=m_total['date_x'].dt.year
m_total['act_month']=m_total['date_x'].dt.month
m_total['act_day']=m_total['date_x'].dt.day
m_total['act_weekday']=m_total['date_x'].dt.weekday
m_total['act_date_int']=(m_total['date_x']-pd.datetime(2020,1,7)).dt.days
m_total['peo_date_int']=(m_total['date_y']-pd.datetime(2020,1,7)).dt.days
######################################
print('product features') 
######################################
m_total['char_10_x_char_38_rt_s']=m_total['char_10_x_char_38_rt']>=0.5
m_total['char_10_x_char_38_rt_us']=m_total['char_10_x_char_38_rt']<0.5
m_total['act_week_num_group_rt_s']=m_total['act_week_num_group_rt']>=0.5
m_total['act_week_num_group_rt_us']=m_total['act_week_num_group_rt']<0.5
m_total['group_rt_s']=m_total['group_rt']>=0.5
m_total['group_rt_us']=m_total['group_rt']<0.5
######################################
print('final data treatment')
######################################
peo_data=simple_data_treat(peo_data)
m_total=simple_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
del m_total
del m_train
######################################
print('train test split')
######################################
#Even after removing the top 5 users, the error rate is still far away from Kaggle error rate. That means we should sample on people_id. As it turns out, this is a much better indicator of real auc. This mean people_id are important.
#An efficient approach to train and test is to separate a small group of peoples as test. Among the remaining people, we can mix their activities. This will be a good mimic of error rate, and will improve our auc.
peo_train=peo_data[peo_data['people_id'].isin(data_train['people_id'])]
peo_train_fraction=peo_train.sample(frac=sample_frac)
t_peo_train,t_peo_test=train_test_split(peo_train_fraction,test_size=0.2)
t_gp_intersect=np.intersect1d(t_peo_train['group_1'].values,t_peo_test['group_1'].values)
t_train_gp=data_train[data_train['people_id'].isin(t_peo_train['people_id']) & data_train['group_1'].isin(t_gp_intersect)]
t_train_no_gp=data_train[data_train['people_id'].isin(t_peo_train['people_id']) & ~(data_train['group_1'].isin(t_gp_intersect))]
t_test_gp=data_train[data_train['people_id'].isin(t_peo_test['people_id']) & data_train['group_1'].isin(t_gp_intersect)]
t_test_no_gp=data_train[data_train['people_id'].isin(t_peo_test['people_id']) & ~(data_train['group_1'].isin(t_gp_intersect))]
#This is to compensate the fact that for some gp, 'act_week_num_group_rt' is not available.
act_id_fraction=t_train_gp['activity_id'].sample(frac=0.25)
t_train_gp_act_week_num_group_rt=t_train_gp['act_week_num_group_rt']
t_train_gp_act_week_num_group_rt_s=t_train_gp['act_week_num_group_rt_s']
t_train_gp_act_week_num_group_rt_us=t_train_gp['act_week_num_group_rt_us']
t_train_gp_act_week_num_group_rt.loc[act_id_fraction.index]=60000
t_train_gp_act_week_num_group_rt_s.loc[act_id_fraction.index]=0
t_train_gp_act_week_num_group_rt_us.loc[act_id_fraction.index]=0
t_train_gp['act_week_num_group_rt']=t_train_gp_act_week_num_group_rt
t_train_gp['act_week_num_group_rt_s']=t_train_gp_act_week_num_group_rt_s
t_train_gp['act_week_num_group_rt_us']=t_train_gp_act_week_num_group_rt_us
del t_train_gp_act_week_num_group_rt
del t_train_gp_act_week_num_group_rt_s
del t_train_gp_act_week_num_group_rt_us

act_id_fraction=t_test_gp['activity_id'].sample(frac=0.2)
t_test_gp_act_week_num_group_rt=t_test_gp['act_week_num_group_rt']
t_test_gp_act_week_num_group_rt_s=t_test_gp['act_week_num_group_rt_s']
t_test_gp_act_week_num_group_rt_us=t_test_gp['act_week_num_group_rt_us']
t_test_gp_act_week_num_group_rt.loc[act_id_fraction.index]=60000
t_test_gp_act_week_num_group_rt_s.loc[act_id_fraction.index]=0
t_test_gp_act_week_num_group_rt_us.loc[act_id_fraction.index]=0
t_test_gp['act_week_num_group_rt']=t_test_gp_act_week_num_group_rt
t_test_gp['act_week_num_group_rt_s']=t_test_gp_act_week_num_group_rt_s
t_test_gp['act_week_num_group_rt_us']=t_test_gp_act_week_num_group_rt_us
del t_test_gp_act_week_num_group_rt
del t_test_gp_act_week_num_group_rt_s
del t_test_gp_act_week_num_group_rt_us

######################################
print('create df for training/testing')
######################################
available_col=['activity_category', 'activity_id', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'date_x', 'outcome', 'people_id', 'char_1_y', 'group_1','char_2_y', 'date_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_3_y_prod_char_7_y','char_3_7_2_rt', 'char_3_7_9_rt', 'act_week_num','act_week_num_group_rt','act_week_num_prod_group_1', 'group_rt', 'peo_last_act', 'is_last_act','act_year', 'act_month', 'act_day', 'act_weekday', 'act_date_int','peo_date_int', 'char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','act_week_num_group_rt_s', 'act_week_num_group_rt_us', 'group_rt_s','group_rt_us']

categorical_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_3_y_prod_char_7_y','act_week_num','act_week_num_prod_group_1','act_year', 'act_month', 'act_day', 'act_weekday']

gp_use_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_3_y_prod_char_7_y','char_3_7_2_rt', 'char_3_7_9_rt', 'act_week_num','act_week_num_group_rt', 'group_rt', 'is_last_act','act_year', 'act_month', 'act_day', 'act_weekday', 'act_date_int','peo_date_int', 'char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','act_week_num_group_rt_s', 'act_week_num_group_rt_us', 'group_rt_s','group_rt_us']

no_gp_use_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_3_y_prod_char_7_y','char_3_7_2_rt', 'char_3_7_9_rt', 'act_week_num','is_last_act','act_year', 'act_month', 'act_day', 'act_weekday', 'act_date_int','peo_date_int', 'char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us']

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
#test_pred=bst.predict(test_dtest)
#act_test2=pd.read_csv('act_test.csv',parse_dates=['date'])
#output = pd.DataFrame({ 'activity_id' : act_test2['activity_id'], 'outcome': test_pred })
#output.to_csv('redhat_pred.csv', index = False)

