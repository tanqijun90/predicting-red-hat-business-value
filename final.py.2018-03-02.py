#super_rt
#We are going to use feature substitute method to deal with missing feature, for example, in places where the group success rate of the current week is not available, we will use the group success rate over time.
#optimize simple data treat
#train test split at the beginning for better code reusability
#We are going to train things together.
######################################
print('parameters')
######################################
kaggle_output=False
sample_frac=0.5
early_stopping_rounds=10
eta=0.1
num_round =500
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
######################################
print('train test split')
######################################
#Even after removing the top 5 users, the error rate is still far away from Kaggle error rate. That means we should sample on people_id. As it turns out, this is a much better indicator of real auc. This mean people_id are important.
#An efficient approach to train and test is to separate a small group of peoples as test. Among the remaining people, we can mix their activities. This will be a good mimic of error rate, and will improve our auc.
if ~ kaggle_output:
    peo_train=peo_data[peo_data['people_id'].isin(act_train['people_id'])]
    peo_train_fraction=peo_train.sample(frac=sample_frac)
    t_peo_train,t_peo_test=train_test_split(peo_train_fraction,test_size=0.2)
    act_test=act_train[act_train['people_id'].isin(t_peo_test['people_id'])]
    act_train=act_train[act_train['people_id'].isin(t_peo_train['people_id'])]
    local_outcome=act_test['outcome']
    del act_test['outcome']
######################################
print('initial data treatment')
######################################
act_train=initial_data_treat(act_train)
act_test=initial_data_treat(act_test)
peo_data=initial_data_treat(peo_data)
act_total=act_train.append(act_test)
m_total=act_total.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
m_train=m_total.iloc[:len(act_train)]
#We create a m_small to exclude the larges four groups in order to reduce the bias they introduce.
m_small=m_train[(m_train['group_1']!=17304) & (m_train['group_1']!=667) & (m_train['group_1']!=8386) &(m_train['group_1']!=9280)]
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
#After using secondary feature, we might want to build a difference between the original feature and the secondary feature.

gpb=m_small.groupby(['char_10_x','char_38'])
char_10_x_char_38_rt=gpb['outcome'].mean()
char_10_x_char_38_rt.name='char_10_x_char_38_rt'
m_total=m_total.merge(pd.DataFrame(char_10_x_char_38_rt),how='left',left_on=['char_10_x','char_38'],right_index=True)
#575k has null char_10_x_char_38_rt. 137257/498687 in test has null char_10_x_char_38_rt

m_total=prod_feature(m_total,'char_6_y','char_7_y')
m_total=prod_feature(m_total,'char_6_y_prod_char_7_y','char_32','super_prod')
m_total=prod_feature(m_total,'super_prod','char_18','super_prod')
m_total=prod_feature(m_total,'super_prod','char_25','super_prod')
m_total=prod_feature(m_total,'super_prod','char_23','super_prod')
m_total=prod_feature(m_total,'super_prod','char_29','super_prod')
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

m_total['is_last_act']=m_total['date_x']==m_total['peo_last_act']
m_total['act_year']=m_total['date_x'].dt.year
m_total['act_month']=m_total['date_x'].dt.month
m_total['act_day']=m_total['date_x'].dt.day
m_total['act_weekday']=m_total['date_x'].dt.weekday
m_total['act_date_int']=(m_total['date_x']-pd.datetime(2020,1,7)).dt.days
m_total['peo_date_int']=(m_total['date_y']-pd.datetime(2020,1,7)).dt.days

del m_total['peo_last_act']
#Product feature might not work here since we are actually dealing with missing features instead of product features. Even using the product features, we are simply moving the data with missing feature to the other half of the devision. Therefore, the RF might not consider it at all.
######################################
print('secondary features')
######################################
#act_week_num_group_rt>group_rt>char_10_x_char_38_rt>super_rt
#13 in test does not have any of these values.
#We will just fill those 13 entries with 0.5
m_total['fill_05']=0.5
m_total=secondary_feature(m_total,'super_rt','fill_05')
m_total=secondary_feature(m_total,'char_10_x_char_38_rt','super_rt')
m_total=secondary_feature(m_total,'group_rt','char_10_x_char_38_rt')
m_total=secondary_feature(m_total,'act_week_num_group_rt','group_rt')
del m_total['fill_05']
m_total['act-gp']=m_total['act_week_num_group_rt']-m_total['group_rt']
m_total['gp-ch']=m_total['group_rt']-m_total['char_10_x_char_38_rt']
m_total['ch-su']=m_total['char_10_x_char_38_rt']-m_total['super_rt']
######################################
print('final data treatment')
######################################
m_total=final_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
######################################
print('train data mimic test data')
######################################
#We shall erase the group_rt and act_week_num_group_rt of certain 20% groups , and then erase the act_week_num_group_rt 20% activities of the remaining groups. We shall also recompute act-gp, gp-ch, ch-su.
t_group_rt=data_train['group_rt']
t_act_week_num_group_rt=data_train['act_week_num_group_rt']
import random
mock_miss_gp=set(random.sample(set(data_train['group_1']),int(len(set(data_train['group_1']))*0.2)))-{17304}
t_group_rt[data_train['group_1'].isin(mock_miss_gp)]=np.nan
t_act_week_num_group_rt[data_train['group_1'].isin(mock_miss_gp)]=np.nan
mock_miss_act=data_train.sample(frac=0.25).index
t_act_week_num_group_rt.loc[mock_miss_act]=np.nan
data_train['group_rt']=t_group_rt
data_train['act_week_num_group_rt']=t_act_week_num_group_rt

data_train=secondary_feature(data_train,'group_rt','char_10_x_char_38_rt')
data_train=secondary_feature(data_train,'act_week_num_group_rt','group_rt')
data_train['act-gp']=data_train['act_week_num_group_rt']-data_train['group_rt']
data_train['gp-ch']=data_train['group_rt']-data_train['char_10_x_char_38_rt']
######################################
print('create df for training/testing')
######################################
available_col=['activity_category', 'activity_id', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'date_x', 'outcome', 'people_id', 'char_1_y', 'group_1','char_2_y', 'date_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_6_y_prod_char_7_y','super_rt', 'act_week_num', 'act_week_num_group_rt', 'group_rt', 'is_last_act', 'act_year','act_month', 'act_day', 'act_weekday', 'act_date_int', 'peo_date_int','act-gp', 'gp-ch', 'ch-su']
categorical_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_6_y_prod_char_7_y', 'act_year','act_month', 'act_day', 'act_weekday']
use_col=['activity_category',  'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1','char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y','char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12','char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18','char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24','char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30','char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36','char_37', 'char_38', 'char_10_x_char_38_rt', 'char_6_y_prod_char_7_y','super_rt', 'act_week_num', 'act_week_num_group_rt', 'group_rt', 'is_last_act', 'act_year','act_month', 'act_day', 'act_weekday', 'act_date_int', 'peo_date_int','act-gp', 'gp-ch', 'ch-su']

df_train=data_train[use_col]
df_test=data_test[use_col]
######################################
print('create OneHot')
######################################
df_categorical_col=list(set(use_col)&set(categorical_col))
df_not_categorical_col=list(set(use_col)-set(df_categorical_col))

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(handle_unknown='ignore')
enc=enc.fit(pd.concat([data_train[df_categorical_col],data_test[df_categorical_col]]))
df_cat_train=enc.transform(df_train[df_categorical_col])
df_cat_test=enc.transform(df_test[df_categorical_col])

from scipy.sparse import hstack
df_spr_train=hstack((df_train[df_not_categorical_col],df_cat_train))
df_spr_test=hstack((df_test[df_not_categorical_col],df_cat_test))
dtrain=xgb.DMatrix(df_spr_train,label=data_train['outcome'])
dtest=xgb.DMatrix(df_spr_test)
######################################
print('run test')
######################################
watchlist  = [(dtrain,'train')]
param = {'max_depth':12, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
######################################
print('output')
######################################
from sklearn.metrics import roc_auc_score
if ~ kaggle_output:
    print(roc_auc_score(local_outcome,bst.predict(dtest)))
else:
    test_pred=bst.predict(dtest)
    output = pd.DataFrame({ 'activity_id' : act_test['activity_id'], 'outcome': test_pred })
    output.to_csv('redhat_pred.csv', index = False)

