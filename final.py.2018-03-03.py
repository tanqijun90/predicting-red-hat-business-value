#secondary feature does not seem to be a good idea.
#revert back to fillna
#revert back to separate training
#use entire data set for wo training, and use overlaping data for gp training

#better mimic of testing data set
#the rates might need to come with a confidence rating
#the leakage in 03-01 might be that there are unrealisticly accurate estimates on the various rates.

#when combining the groups together, we can shrink the no_gp group to around 0.5 by some factor to indicate different confidence level of our models
#use interpolation

######################################
print('parameters')
######################################
kaggle_output=False
sample_frac=0.5
early_stopping_rounds=10
eta=0.1
num_round_gp =100
num_round_wo =40
tree_build_subsample=1
col_sample_tree=1
######################################
print('functions')
######################################
def prod_feature(data,col1,col2,new_col_name='no_name_provided'):
    if new_col_name=='no_name_provided':
        new_col_name=col1+'_prod_'+col2
    gpb=data.groupby(by=[col1,col2])
    new_index=gpb[col1].count()
    list_index=list(range(len(new_index)))
    new_index[:]=list_index
    new_index.name=new_col_name+'_new'
    new_index.astype('int32')
    data=data.merge(pd.DataFrame(new_index),how='left',left_on=[col1,col2],right_index=True)
    if new_col_name in data.columns:
        del data[new_col_name]
    data.rename(columns={new_col_name+'_new':new_col_name},inplace=True)
    return data

def secondary_feature(data,col_p,col_s):
    col_prim=data[col_p]
    col_secd=data[col_s]
    col_prim[col_prim.isnull()]=col_secd[col_prim.isnull()]
    data[col_p]=col_prim
    return data

def initial_data_treat(data):
    data.replace(to_replace='.* ',regex=True,value='',inplace=True)
    for col in data.columns:
        if (data[col].dtype=='O') & (col!='people_id') & (col!='activity_id'):
            data[col]=pd.to_numeric(data[col],errors='coerce')
            data[col]=data[col].astype('int32',errors='ignore')
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    return data

def final_data_treat(data):
    for col in data.columns:
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    for col in (set(['char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x']) & set(data.columns)):
        data[col].fillna(60000,inplace=True)#fill number cannot be negative, cannot be too large
        data[col]=data[col].astype('int32',errors='ignore')
    for col in (set(['char_10_x_char_38_rt','super_rt', 'act_week_num_group_rt', 'group_rt']) & set(data.columns)):
        data[col].fillna(60000,inplace=True)#fill number cannot be negative, cannot be too large
    return data
#Is there anything else to fillin?

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
######################################
print('train test split')
######################################
#Even after removing the top 5 users, the error rate is still far away from Kaggle error rate. That means we should sample on people_id. As it turns out, this is a much better indicator of real auc. This mean people_id are important.
#An efficient approach to train and test is to separate a small group of peoples as test. Among the remaining people, we can mix their activities. This will be a good mimic of error rate, and will improve our auc.
if not kaggle_output:
    peo_train=peo_data[peo_data['people_id'].isin(act_train['people_id'])]
    peo_train_fraction=peo_train.sample(frac=sample_frac)
    t_peo_train,t_peo_test=train_test_split(peo_train_fraction,test_size=0.2)
    act_test=act_train[act_train['people_id'].isin(t_peo_test['people_id'])]
    act_train=act_train[act_train['people_id'].isin(t_peo_train['people_id'])]
    local_outcome=act_test[['activity_id','outcome']]
    del act_test['outcome']
######################################
print('initial data treatment')
######################################
act_train=initial_data_treat(act_train)
act_test=initial_data_treat(act_test)
peo_data=initial_data_treat(peo_data)
act_total=act_train.append(act_test)
m_total=act_total.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
#############################################
print('feature engineer')
#############################################
#We need to engineer 
#char_10_x char_38 success rate
#char_6_y char_7_y product
#super_rt
#act_week_num
#act_week_num group_1 success rate
#group_rt
#act_weekday,act_year,act_month,act_day
#is_last_act
#convert date_x,date_y into integers
#We also need to pay attension to the various success rate for t_test, since some of them should not be available.
#When computing success rates that are not group related, we use m_small.

m_train=m_total.iloc[:len(act_train)]
#We create a m_small to exclude the larges four groups in order to reduce the bias they introduce.
m_small=m_train[(m_train['group_1']!=17304) & (m_train['group_1']!=667) & (m_train['group_1']!=8386) &(m_train['group_1']!=9280)]
gpb=m_small.groupby(['char_10_x','char_38'])
char_10_x_char_38_rt=gpb['outcome'].mean()
char_10_x_char_38_rt.name='char_10_x_char_38_rt'
m_total=m_total.merge(pd.DataFrame(char_10_x_char_38_rt),how='left',left_on=['char_10_x','char_38'],right_index=True)
#575k has null char_10_x_char_38_rt. 137257/498687 in test has null char_10_x_char_38_rt

m_total=prod_feature(m_total,'char_6_y','char_7_y')
m_total=prod_feature(m_total,'char_6_y_prod_char_7_y','char_32','super_prod')
m_total=prod_feature(m_total,'super_prod','char_18','super_prod')
#m_total=prod_feature(m_total,'super_prod','char_25','super_prod')
#m_total=prod_feature(m_total,'super_prod','char_23','super_prod')
#m_total=prod_feature(m_total,'super_prod','char_29','super_prod')
m_train=m_total.iloc[:len(act_train)]
m_small=m_train[(m_train['group_1']!=17304) & (m_train['group_1']!=667) & (m_train['group_1']!=8386) &(m_train['group_1']!=9280)]
gpb=m_small.groupby('super_prod')
super_rt=gpb['outcome'].mean()
super_rt.name='super_rt'
m_total=m_total.merge(pd.DataFrame(super_rt),how='left',left_on='super_prod',right_index=True)
del m_total['super_prod']
del m_small
#7446 has null super_rt. 1659 in test has null super_rt

m_total['act_week_num']=np.floor(((m_total['date_x']-pd.datetime(2020,1,7)).dt.days)/7)
m_total['act_week_num']=m_total['act_week_num'].astype('int32')
m_train=m_total.iloc[:len(act_train)]
gpb=m_train.groupby(['act_week_num','group_1'])
act_week_num_group_rt=gpb['outcome'].mean()
act_week_num_group_rt.name='act_week_num_group_rt'
m_total=m_total.merge(pd.DataFrame(act_week_num_group_rt),how='left',left_on=['act_week_num','group_1'],right_index=True)

gpb=m_train.groupby(['group_1'])
group_rt=gpb['outcome'].mean()
group_rt.name='group_rt'
m_total=m_total.merge(pd.DataFrame(group_rt),how='left',left_on=['group_1'],right_index=True)
del m_train

peo=m_total.groupby('people_id')
peo_last_act=peo['date_x'].max()
peo_last_act.name='peo_last_act'
m_total=m_total.merge(pd.DataFrame(peo_last_act),how='left',left_on=['people_id'],right_index=True)

m_total['is_last_act']=(m_total['date_x']==m_total['peo_last_act'])
m_total['act_year']=m_total['date_x'].dt.year
m_total['act_month']=m_total['date_x'].dt.month
m_total['act_day']=m_total['date_x'].dt.day
m_total['act_weekday']=m_total['date_x'].dt.weekday
m_total['act_date_int']=(m_total['date_x']-pd.datetime(2020,1,7)).dt.days
m_total['peo_date_int']=(m_total['date_y']-pd.datetime(2020,1,7)).dt.days

del m_total['peo_last_act']
######################################
print('product features') 
######################################
m_total['super_rt_s']=m_total['super_rt']>=0.5
m_total['super_rt_us']=m_total['super_rt']>=0.5
m_total['char_10_x_char_38_rt_s']=m_total['char_10_x_char_38_rt']>=0.5
m_total['char_10_x_char_38_rt_us']=m_total['char_10_x_char_38_rt']<0.5
m_total['act_week_num_group_rt_s']=m_total['act_week_num_group_rt']>=0.5
m_total['act_week_num_group_rt_us']=m_total['act_week_num_group_rt']<0.5
m_total['group_rt_s']=m_total['group_rt']>=0.5
m_total['group_rt_us']=m_total['group_rt']<0.5
#Product feature might not work here since we are actually dealing with missing features instead of product features. Even using the product features, we are simply moving the data with missing feature to the other half of the devision. Therefore, the RF might not consider it at all.
######################################
print('final data treatment')
######################################
m_total=final_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
######################################
print('create df for training/testing')
######################################
available_col=['activity_category', 'activity_id', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'date_x', 'outcome', 'people_id', 'char_1_y', 'group_1','char_2_y', 'date_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_6_y_prod_char_7_y','super_rt', 'act_week_num', 'act_week_num_group_rt', 'group_rt','is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_date_int', 'peo_date_int', 'super_rt_s', 'super_rt_us','char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','act_week_num_group_rt_s', 'act_week_num_group_rt_us', 'group_rt_s','group_rt_us']
categorical_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_6_y_prod_char_7_y','act_week_num', 'act_year', 'act_month', 'act_day', 'act_weekday']#'char_38',
use_col_gp=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_6_y_prod_char_7_y','super_rt', 'act_week_num', 'act_week_num_group_rt', 'group_rt','is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_date_int', 'peo_date_int', 'super_rt_s', 'super_rt_us','char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','act_week_num_group_rt_s', 'act_week_num_group_rt_us', 'group_rt_s','group_rt_us']
use_col_wo=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_6_y_prod_char_7_y','super_rt', 'act_week_num','is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_date_int', 'peo_date_int', 'super_rt_s', 'super_rt_us','char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us']
df_train_wo=data_train[use_col_wo]
#df_train_gp=data_train[use_col_gp]
######################################
print('train_gp shrink and mimic test data')
######################################
#The leakage in 03-01 might be that the testing group get incredibly accurate act_week_num_group_rt and group_rt. The problem with the current model is that it assumes the leakage and train the model this way. Therefore, the training ends very fast. We need a better emulation of what will happen in the testing set
gp_intersect=np.intersect1d(data_train['group_1'].values,data_test['group_1'].values)
df_train_gp=data_train[data_train['group_1'].isin(gp_intersect)]
train_gp_outcome=df_train_gp['outcome']
df_train_gp=df_train_gp[use_col_gp]# this is a refresh of df_train
#We shall erase the act_week_num_group_rt 20% activities of the remaining groups.
t_act_week_num_group_rt=df_train_gp['act_week_num_group_rt']
t_act_week_num_group_rt_s=df_train_gp['act_week_num_group_rt_s']
t_act_week_num_group_rt_us=df_train_gp['act_week_num_group_rt_us']
mock_miss_act=df_train_gp.sample(frac=0.25).index
t_act_week_num_group_rt.loc[mock_miss_act]=60000
t_act_week_num_group_rt_s.loc[mock_miss_act]=0
t_act_week_num_group_rt_us.loc[mock_miss_act]=0
df_train_gp['act_week_num_group_rt']=t_act_week_num_group_rt
df_train_gp['act_week_num_group_rt_s']=t_act_week_num_group_rt_s
df_train_gp['act_week_num_group_rt_us']=t_act_week_num_group_rt_us
######################################
print('create OneHot')
######################################
df_categorical_col_gp=list(set(use_col_gp)&set(categorical_col))
df_not_categorical_col_gp=list(set(use_col_gp)-set(categorical_col))
df_categorical_col_wo=list(set(use_col_wo)&set(categorical_col))
df_not_categorical_col_wo=list(set(use_col_wo)-set(categorical_col))

from sklearn.preprocessing import OneHotEncoder
enc_gp=OneHotEncoder(handle_unknown='ignore')
enc_gp=enc_gp.fit(pd.concat([df_train_gp[df_categorical_col_gp],data_test[df_categorical_col_gp]]))
enc_wo=OneHotEncoder(handle_unknown='ignore')
enc_wo=enc_wo.fit(pd.concat([df_train_wo[df_categorical_col_wo],data_test[df_categorical_col_wo]]))
df_cat_train_gp=enc_gp.transform(df_train_gp[df_categorical_col_gp])
df_cat_train_wo=enc_wo.transform(df_train_wo[df_categorical_col_wo])

from scipy.sparse import hstack
df_spr_train_gp=hstack((df_train_gp[df_not_categorical_col_gp],df_cat_train_gp))
df_spr_train_wo=hstack((df_train_wo[df_not_categorical_col_wo],df_cat_train_wo))
dtrain_gp=xgb.DMatrix(df_spr_train_gp,label=train_gp_outcome)
dtrain_wo=xgb.DMatrix(df_spr_train_wo,label=data_train['outcome'])
#data_train is not a problem here since we are using the entire set for training
######################################
print('prepare test group')
######################################
df_test_gp=data_test[data_test['group_1'].isin(gp_intersect)]
df_test_wo=data_test[~data_test['group_1'].isin(gp_intersect)]
if not kaggle_output:
    df_test_gp=df_test_gp.merge(local_outcome,how='left',left_on='activity_id',right_on='activity_id')
    df_test_wo=df_test_wo.merge(local_outcome,how='left',left_on='activity_id',right_on='activity_id')
    local_outcome_gp=df_test_gp['outcome_y']
    local_outcome_wo=df_test_wo['outcome_y']
df_test_gp=df_test_gp[use_col_gp]
df_test_wo=df_test_wo[use_col_wo]
df_cat_test_gp=enc_gp.transform(df_test_gp[df_categorical_col_gp])
df_cat_test_wo=enc_wo.transform(df_test_wo[df_categorical_col_wo])
df_spr_test_gp=hstack((df_test_gp[df_not_categorical_col_gp],df_cat_test_gp))
df_spr_test_wo=hstack((df_test_wo[df_not_categorical_col_wo],df_cat_test_wo))
dtest_gp=xgb.DMatrix(df_spr_test_gp)
dtest_wo=xgb.DMatrix(df_spr_test_wo)
######################################
print('run test')
######################################
watchlist  = [(dtrain_gp,'train')]
param = {'max_depth':10, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst_gp = xgb.train(param, dtrain_gp, num_round_gp, watchlist,early_stopping_rounds=early_stopping_rounds)

watchlist  = [(dtrain_wo,'train')]
param = {'max_depth':8, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst_wo = xgb.train(param, dtrain_wo, num_round_wo, watchlist,early_stopping_rounds=early_stopping_rounds)
######################################
print('output')
######################################
from sklearn.metrics import roc_auc_score
if not kaggle_output:
    print(roc_auc_score(local_outcome_gp,bst_gp.predict(dtest_gp)))
    print(roc_auc_score(local_outcome_wo,bst_wo.predict(dtest_wo)))
    print(roc_auc_score(np.concatenate((local_outcome_gp,local_outcome_wo)),np.concatenate((bst_gp.predict(dtest_gp),bst_wo.predict(dtest_wo)))))
else:
    pred_gp=bst_gp.predict(dtest_gp)
    pred_wo=bst_wo.predict(dtest_wo)
    act_id_gp=data_test[data_test['group_1'].isin(gp_intersect)]['activity_id']
    act_id_wo=data_test[~ data_test['group_1'].isin(gp_intersect)]['activity_id']
    output_gp = pd.DataFrame({ 'activity_id' : act_id_gp, 'outcome': pred_gp })
    output_wo = pd.DataFrame({ 'activity_id' : act_id_wo, 'outcome': pred_wo })
    output=output_gp.append(output_wo)
    output.to_csv('redhat_pred.csv', index = False)

