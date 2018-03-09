#train together
######################################
print('parameters')
######################################
fillna_num=60000#fill number cannot be negative, cannot be too large
kaggle_output=False
confidence_wo=1
sample_frac=1
train_frac=0.2
train_wo_frac=0.05
early_stopping_rounds=10
eta=0.02
num_round =200
tree_build_subsample=1
col_sample_tree=1
small_group_act_size=100
######################################
print('functions')
######################################
def interpolate_funct(x,x_0,x_1,y_0,y_1,alpha=4):
    l=2*(x-x_0)/(x_1-x_0)-1 #l(x_1)=1,l(x_0)=-1
    phi=np.tanh(alpha*l)/np.tanh(alpha)#phi(1)=1,phi(-1)=-1
    return (y_1-y_0)*phi/2+(y_1+y_0)/2

def interpolate_rt(df_sorted_by_date,col_of_date_int,col_of_average,date_int,fillna_num=60000):
    if date_int > df_sorted_by_date[col_of_date_int].iloc[-1] or date_int < df_sorted_by_date[col_of_date_int].iloc[0]:
        return fillna_num
    ind=df_sorted_by_date[col_of_date_int].searchsorted(date_int)[0]
    if date_int==df_sorted_by_date[col_of_date_int].iloc[ind]:
        return df_sorted_by_date[col_of_average].iloc[ind]
    return interpolate_funct(date_int,df_sorted_by_date[col_of_date_int].iloc[ind-1],df_sorted_by_date[col_of_date_int].iloc[ind],df_sorted_by_date[col_of_average].iloc[ind-1],df_sorted_by_date[col_of_average].iloc[ind])

def interpolate_rt_wrapped(df_to_interpolate,gpb_sorted,col_of_date_int,col_of_average):
    rt=df_to_interpolate.apply(lambda entry : interpolate_rt(gpb_sorted.loc[entry[0]],col_of_date_int,col_of_average,entry[1]),axis=1)
    return rt

def interpolate_rt_df(df_to_interpolate,df_data,group_by,col_of_date_int,col_of_average):
#df_to_interpolate needs to have two cols, with group_by being the fisrt and col_of_date_int the second
#the function will destroy df_to_interpolate
    gpb=df_data.groupby(group_by)
    gpb_sorted=gpb[[col_of_date_int,col_of_average]].apply(partial(pd.DataFrame.sort_values,by=col_of_date_int))
    df_to_interpolate.drop_duplicates(inplace=True)
    with Pool(6) as p:
        df_to_interpolate_split=np.array_split(df_to_interpolate,6)
        df_to_interpolate[col_of_average]=pd.concat(p.map(partial(interpolate_rt_wrapped,gpb_sorted=gpb_sorted,col_of_date_int=col_of_date_int,col_of_average=col_of_average),df_to_interpolate_split))
    return df_to_interpolate
    
def success_rate(data_total,cols_list,data_use=[],fillna_num=60000):
    if len(data_use)==0:
        data_use=data_total
    gpb=data_use.groupby(cols_list)
    cols_rt=gpb['outcome'].mean()
    col_name=''
    for col in cols_list:
        col_name=col_name+col+'_'
    cols_rt.name=col_name+'rt'
    data_total=data_total.merge(pd.DataFrame(cols_rt),how='left',left_on=cols_list,right_index=True)
    data_total[cols_rt.name].fillna(fillna_num,inplace=True)
    return data_total

def unmask_rt(data_total,col_rt,col_over=[],fillna_num=60000):
    data_total[col_rt+'_s']=(data_total[col_rt]>=0.5) &(data_total[col_rt]<fillna_num)
    data_total[col_rt+'_us']=data_total[col_rt]<0.5
    for col in col_over:
        data_total[col_rt+'_s']=data_total[col_rt+'_s']&data_total[col].isnull()
        data_total[col_rt+'_us']=data_total[col_rt+'_us']&data_total[col].isnull()
    data_total[col_rt+'_s']=data_total[col_rt+'_s'].astype('int8')
    data_total[col_rt+'_us']=data_total[col_rt+'_us'].astype('int8')
    return data_total

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
        if (data[col].dtype=='O') and (col!='people_id') and (col!='activity_id'):
            data[col]=pd.to_numeric(data[col],errors='coerce')
            data[col]=data[col].astype('int32',errors='ignore')
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    return data

def final_data_treat(data,fillna_num=60000):
    for col in data.columns:
        if (data[col].dtype=='bool'):
            data[col]=data[col].astype('int8')
    for col in (set(['char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x']) & set(data.columns)):
        data[col].fillna(fillna_num,inplace=True)
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
gp_intersect=np.intersect1d(m_total['group_1'].iloc[:len(act_train)].values,m_total['group_1'].iloc[len(act_train):].values)
#############################################
print('date-related feature engineering')
#############################################
peo=m_total.groupby('people_id')
peo_last_act=peo['date_x'].max()
peo_last_act.name='peo_last_act'
m_total=m_total.merge(pd.DataFrame(peo_last_act),how='left',left_on=['people_id'],right_index=True)
m_total['is_last_act']=(m_total['date_x']==m_total['peo_last_act'])
m_total['act_year']=m_total['date_x'].dt.year
m_total['act_month']=m_total['date_x'].dt.month
m_total['act_day']=m_total['date_x'].dt.day
m_total['act_weekday']=m_total['date_x'].dt.weekday
m_total['act_week_num']=np.floor(((m_total['date_x']-pd.datetime(2020,1,7)).dt.days)/7)
m_total['act_week_num']=m_total['act_week_num'].astype('int32')
m_total['act_date_int']=(m_total['date_x']-pd.datetime(2020,1,7)).dt.days
m_total['peo_date_int']=(m_total['date_y']-pd.datetime(2020,1,7)).dt.days
del m_total['peo_last_act']
del m_total['date_x']
del m_total['date_y']
#############################################
print('digression to group_act_size')
#############################################
gpb=m_total.groupby('group_1')
t_group_act_size=gpb['activity_id'].count()
t_group_act_size.name='group_act_size'
m_total=m_total.merge(pd.DataFrame(t_group_act_size),how='left',left_on='group_1',right_index=True)
#############################################
print('product and success rate feature engineering')
#############################################
#We also need to pay attension to the various success rate for t_test, since some of them should not be available.
m_total=prod_feature(m_total,'char_6_y','char_7_y')
m_train=m_total.iloc[:len(act_train)]
#We create a m_small to exclude the larges four groups in order to reduce the bias they introduce.
#When computing success rates that are not group related, we use m_small.
m_small=m_train[m_train['group_act_size']<=small_group_act_size]
m_total=success_rate(m_total,['char_10_x','char_38'],m_small)
#575k has null char_10_x_char_38_rt. 137257/498687 in test has null char_10_x_char_38_rt
m_total=success_rate(m_total,['char_6_y','char_7_y','char_32','char_18'],m_small)
del m_small
#7446 has null super_rt. 1659 in test has null super_rt
m_total=success_rate(m_total,['group_1'],m_train)
m_total=success_rate(m_total,['act_date_int','group_1'],m_train)
del m_train
#############################################
print('group date interpolation')
#############################################
from functools import partial
m_train=m_total.iloc[:len(act_train)]
m_test=m_total.iloc[len(act_train):]
m_test_gp=m_test[m_test['group_1'].isin(gp_intersect)]
m_test_gp=m_test_gp[['group_1','act_date_int']]
m_test_gp=interpolate_rt_df(m_test_gp,m_train,'group_1','act_date_int','act_date_int_group_1_rt')
del m_test['act_date_int_group_1_rt']
m_test=m_test.merge(m_test_gp,how='left',on=['group_1','act_date_int'])
m_test['act_date_int_group_1_rt'].fillna(fillna_num,inplace=True)
del m_total['act_date_int_group_1_rt']
m_total['act_date_int_group_1_rt']=pd.concat([m_train['act_date_int_group_1_rt'],m_test['act_date_int_group_1_rt']]).reset_index(drop=True)
del m_test
del m_test_gp
######################################
print('unmasking features') 
######################################
m_total=unmask_rt(m_total,'act_date_int_group_1_rt')
m_total=unmask_rt(m_total,'group_1_rt',['act_date_int_group_1_rt'])
m_total=unmask_rt(m_total,'char_10_x_char_38_rt')
m_total=unmask_rt(m_total,'char_6_y_char_7_y_char_32_char_18_rt',['char_10_x_char_38_rt'])
######################################
print('final data treatment')
######################################
m_total['fill05']=0.5
m_total=final_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
######################################
print('columns for training/testing')
######################################
available_col=['activity_category', 'activity_id', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'outcome', 'people_id', 'char_1_y', 'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38','is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_week_num', 'act_date_int', 'peo_date_int','char_6_y_prod_char_7_y','group_act_size', 'char_10_x_char_38_rt','char_6_y_char_7_y_char_32_char_18_rt', 'group_1_rt','act_date_int_group_1_rt', 'act_date_int_group_1_rt_s','act_date_int_group_1_rt_us', 'group_1_rt_s', 'group_1_rt_us','char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','char_6_y_char_7_y_char_32_char_18_rt_s','char_6_y_char_7_y_char_32_char_18_rt_us']

categorical_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'act_year', 'act_month', 'act_day', 'act_weekday','act_week_num', 'char_6_y_prod_char_7_y']

use_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38','is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_week_num', 'act_date_int', 'peo_date_int','char_6_y_prod_char_7_y','act_date_int_group_1_rt',  'act_date_int_group_1_rt_s','act_date_int_group_1_rt_us', 'group_1_rt_s', 'group_1_rt_us','char_10_x_char_38_rt','char_6_y_char_7_y_char_32_char_18_rt', 'group_1_rt','char_10_x_char_38_rt_s', 'char_10_x_char_38_rt_us','char_6_y_char_7_y_char_32_char_18_rt_s','char_6_y_char_7_y_char_32_char_18_rt_us']
######################################
print('df_train water down')
######################################
#The leakage in 03-01 might be that the testing group get incredibly accurate act_week_num_group_rt and group_rt. The problem with the current model is that it assumes the leakage and train the model this way. Therefore, the training ends very fast. We need a better emulation of what will happen in the testing set
#We will simply drop some act_date_int_group_1_rt instead of retraining because the values are so accurate when they do exist.
df_train=data_train
df_train.rename(columns={'act_date_int_group_1_rt':'act_date_int_group_1_rt_optimistic'},inplace=True)
df_train.rename(columns={'group_1_rt':'group_1_rt_optimistic'},inplace=True)
del df_train['act_date_int_group_1_rt_s']
del df_train['act_date_int_group_1_rt_us']
del df_train['group_1_rt_s']
del df_train['group_1_rt_us']
mock_miss_act=df_train.sample(frac=train_frac).index
t_df_train_frac=df_train.loc[mock_miss_act]
df_train=success_rate(df_train,['group_1'],t_df_train_frac)
t_act_date_int_group_1_rt=df_train['act_date_int_group_1_rt_optimistic']
t_act_date_int_group_1_rt[(df_train['group_act_size']<=small_group_act_size)& t_act_date_int_group_1_rt.index.isin(t_act_date_int_group_1_rt.sample(frac=1).index)]=fillna_num
t_act_date_int_group_1_rt[t_act_date_int_group_1_rt.index.isin(t_act_date_int_group_1_rt.sample(frac=0.24).index)]=fillna_num
df_train['act_date_int_group_1_rt']=t_act_date_int_group_1_rt
del t_act_date_int_group_1_rt
del t_df_train_frac
df_train=secondary_feature(df_train,'group_1_rt','fill05')
df_train=unmask_rt(df_train,'act_date_int_group_1_rt')
df_train=unmask_rt(df_train,'group_1_rt',['act_date_int_group_1_rt'])
df_train=df_train[use_col]
######################################
print('create OneHot')
######################################
df_categorical_col=list(set(use_col)&set(categorical_col))
df_not_categorical_col=list(set(use_col)-set(categorical_col))
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(handle_unknown='ignore')
enc=enc.fit(pd.concat([df_train[df_categorical_col],data_test[df_categorical_col]]))
df_cat_train=enc.transform(df_train[df_categorical_col])
from scipy.sparse import hstack
df_spr_train=hstack((df_train[df_not_categorical_col],df_cat_train))
dtrain=xgb.DMatrix(df_spr_train,label=data_train['outcome'])
######################################
print('prepare test group')
######################################
df_test=data_test
df_test=df_test[use_col]
df_cat_test=enc.transform(df_test[df_categorical_col])
df_spr_test=hstack((df_test[df_not_categorical_col],df_cat_test))
dtest=xgb.DMatrix(df_spr_test)
######################################
print('run test')
######################################
watchlist  = [(dtrain,'train')]
param = {'max_depth':8, 'eta':eta, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
######################################
print('output')
######################################
from sklearn.metrics import roc_auc_score
if not kaggle_output:
    print(roc_auc_score(local_outcome['outcome'],bst.predict(dtest)))
else:
    output = pd.DataFrame({ 'activity_id' : data_test['activity_id'], 'outcome': bst.predict(dtest)})
    output.to_csv('redhat_pred.csv', index = False)
