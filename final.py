######################################
print('parameters')
######################################
fillna_num=60000#fill number cannot be negative, cannot be too large
kaggle_output=True
confidence_wo=0.2
alpha=1
sample_frac=1
train_gp_frac=0.2# we can play with this number
train_wo_frac=0.1
early_stopping_rounds=10
eta_gp=0.1
eta_wo=0.02
num_round_gp =60
num_round_wo =500
tree_build_subsample=1
col_sample_tree=1
small_group_act_size=10000000
######################################
print('functions')
######################################
def prediction_mod_funct(prediction,alpha,confidence_wo):
    return confidence_wo*(prediction-0.5)+0.5

def interpolate_funct(x,x_0,x_1,y_0,y_1,center=0.5,alpha=0.5):#alpha does not change gp_AUC
    l=2*(x-x_0)/(x_1-x_0)-1 #l(x_1)=1,l(x_0)=-1
    phi=np.tanh(alpha*l)/np.tanh(alpha)#phi(1)=1,phi(-1)=-1
    return (y_1-y_0)*phi/2+(y_1+y_0)/2

def interpolate_funct_boundary(x,x_0,y_0,boundary,decay,alpha=0.00001,alpha_limit=1):
#    boundary_limit=(y_0+decay*alpha_limit*boundary)/(1+decay*alpha_limit*boundary)
    if y_0==1:
        boundary_limit=0.95
    else:
        boundary_limit=0.05
#    return boundary_limit
    return (y_0-boundary_limit)*np.exp(-alpha*decay*abs(x-x_0))+boundary_limit

def interpolate_rt(df_sorted_by_date,col_of_date_int,col_of_average,col_of_boundary,col_of_decay,interpolate_boundary,date_int,fillna_num=60000):
    if interpolate_boundary:
        if date_int > df_sorted_by_date[col_of_date_int].iloc[-1]:
            return interpolate_funct_boundary(date_int,df_sorted_by_date[col_of_date_int].iloc[-1],df_sorted_by_date[col_of_average].iloc[-1],df_sorted_by_date[col_of_boundary].iloc[-1],df_sorted_by_date[col_of_decay].iloc[-1])
        if date_int < df_sorted_by_date[col_of_date_int].iloc[0]:
            return interpolate_funct_boundary(date_int,df_sorted_by_date[col_of_date_int].iloc[0],df_sorted_by_date[col_of_average].iloc[0],df_sorted_by_date[col_of_boundary].iloc[0],df_sorted_by_date[col_of_decay].iloc[0])
        return fillna_num
    else:
        if date_int > df_sorted_by_date[col_of_date_int].iloc[-1]:
            return fillna_num
        if date_int < df_sorted_by_date[col_of_date_int].iloc[0]:
            return fillna_num
        ind=df_sorted_by_date[col_of_date_int].searchsorted(date_int)[0]
        if date_int==df_sorted_by_date[col_of_date_int].iloc[ind]:
            return df_sorted_by_date[col_of_average].iloc[ind]
        return interpolate_funct(date_int,df_sorted_by_date[col_of_date_int].iloc[ind-1],df_sorted_by_date[col_of_date_int].iloc[ind],df_sorted_by_date[col_of_average].iloc[ind-1],df_sorted_by_date[col_of_average].iloc[ind])

def interpolate_rt_wrapped(df_to_interpolate,gpb_sorted,col_of_date_int,col_of_average,col_of_boundary,col_of_decay,interpolate_boundary):
    rt=df_to_interpolate.apply(lambda entry : interpolate_rt(gpb_sorted.loc[entry[0]],col_of_date_int,col_of_average,col_of_boundary,col_of_decay,interpolate_boundary,entry[1]),axis=1)
    return rt

def interpolate_rt_df(df_to_interpolate,df_data,group_by,col_of_date_int,col_of_average,col_of_boundary,col_of_decay,interpolate_boundary=False):
    from functools import partial
    gp_intersect=np.intersect1d(df_to_interpolate[group_by].values,df_data[group_by].values)
    df_to_interpolate_gp=df_to_interpolate[df_to_interpolate[group_by].isin(gp_intersect)]
    df_to_interpolate_gp=df_to_interpolate_gp[[group_by,col_of_date_int]]
    gpb=df_data.groupby(group_by)
    gpb_sorted=gpb[[col_of_date_int,col_of_average,col_of_boundary,col_of_decay]].apply(partial(pd.DataFrame.sort_values,by=col_of_date_int))
    df_to_interpolate_gp.drop_duplicates(inplace=True)
    with Pool(6) as p:
        df_to_interpolate_gp_split=np.array_split(df_to_interpolate_gp,6)
        df_to_interpolate_gp[col_of_average]=pd.concat(p.map(partial(interpolate_rt_wrapped,gpb_sorted=gpb_sorted,col_of_date_int=col_of_date_int,col_of_average=col_of_average,col_of_boundary=col_of_boundary,col_of_decay=col_of_decay,interpolate_boundary=interpolate_boundary),df_to_interpolate_gp_split))
    if interpolate_boundary:
        df_to_interpolate_gp.rename(columns={col_of_average:col_of_average+'_boundary'},inplace=True)
        col_of_average=col_of_average+'_boundary'
    if col_of_average in df_to_interpolate.columns:
        del df_to_interpolate[col_of_average]
    df_to_interpolate=df_to_interpolate.merge(df_to_interpolate_gp,how='left',on=[group_by,col_of_date_int])
    df_to_interpolate[col_of_average].fillna(fillna_num,inplace=True)#since we are not just using df_to_interpolate_gp
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

def secondary_feature(data,col_p,col_s,fillna_num=60000):
    col_prim=data[col_p]
    col_secd=data[col_s]
    col_prim[col_prim.isnull()]=col_secd[col_prim.isnull()]
    col_prim[col_prim==fillna_num]=col_secd[col_prim==fillna_num]
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

def hist_bin(data,col,bin_size):
    hist_b=data[col].value_counts()
    hist=((hist_b.cumsum()-hist_b/2)/hist_b.sum()*bin_size).apply(np.ceil).astype('int32')
    hist.name=col+'_bin'
    print('{}{}'.format('Actual bin number: ',len(np.unique(hist.values))))
    return data.merge(pd.DataFrame(hist),how='left',left_on=col,right_index=True)

def congregate_small_category(data,col,max_num_corr):
    hist=data[col].value_counts()
    ind=pd.Series(range(len(hist)))
    ind.index=hist.index
    ind.name=col+'_con'
    for i in range(1,max_num_corr+1):
        ind[hist==i]=len(hist)+i
    ind=ind.astype('int32')
    return data.merge(pd.DataFrame(ind),how='left',left_on=col,right_index=True)
######################################
print('read in data')
######################################
act_train=pd.read_csv('../act_train.csv',parse_dates=['date'])
act_test=pd.read_csv('../act_test.csv',parse_dates=['date'])
peo_data=pd.read_csv('../people.csv',parse_dates=['date'])
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
    test_outcome=act_test[['activity_id','outcome']]
    del act_test['outcome']
######################################
print('initial data treatment')
######################################
act_train=initial_data_treat(act_train)
act_test=initial_data_treat(act_test)
peo_data=initial_data_treat(peo_data)
act_total=act_train.append(act_test)
#congregate by people
peo_data=congregate_small_category(peo_data,'group_1',1)
m_total=act_total.merge(peo_data,how='left',on='people_id')
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
m_total['is_weekend']=m_total['act_weekday']>=5
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
del t_group_act_size
#############################################
print('product and success rate feature engineering')
#############################################
#We also need to pay attension to the various success rate for t_test, since some of them should not be available.
#We create a m_small to exclude the larges four groups in order to reduce the bias they introduce.
#When computing success rates that are not group related, we use m_small.
#m_small=m_train[m_train['group_act_size']<=small_group_act_size]
#del m_small

#for test_gp
m_total=prod_feature(m_total,'char_6_y','char_2_y')
#for test_wo
m_total=prod_feature(m_total,'char_5_y','char_6_y')
m_total=prod_feature(m_total,'char_7_y','char_9_y')
m_total=prod_feature(m_total,'char_1_y','char_8_y')

#for test_gp
m_train=m_total.iloc[:len(act_train)]
m_total=success_rate(m_total,['act_date_int','group_1'],m_train)
m_total=success_rate(m_total,['char_2_y','char_6_y','char_7_y','char_9_y'],m_train)
m_total=success_rate(m_total,['group_1'],m_train)

del m_train
#############################################
print('group date interpolation')
#############################################
m_train=m_total.iloc[:len(act_train)]
m_test=m_total.iloc[len(act_train):]

m_test=interpolate_rt_df(m_test,m_train,'group_1','act_date_int','act_date_int_group_1_rt','group_1_rt','group_act_size')
del m_total['act_date_int_group_1_rt']
m_total['act_date_int_group_1_rt']=pd.concat([m_train['act_date_int_group_1_rt'],m_test['act_date_int_group_1_rt']]).reset_index(drop=True)

m_test=interpolate_rt_df(m_test,m_train,'group_1','act_date_int','act_date_int_group_1_rt','group_1_rt','group_act_size',True)
m_total['act_date_int_group_1_rt_boundary']=pd.concat([m_train['act_date_int_group_1_rt'],m_test['act_date_int_group_1_rt_boundary']]).reset_index(drop=True)
del m_train
del m_test
######################################
print('unmasking features') 
######################################
m_total=unmask_rt(m_total,'group_1_rt',['act_date_int_group_1_rt'])
m_total=unmask_rt(m_total,'act_date_int_group_1_rt_boundary',['act_date_int_group_1_rt'])
m_total=unmask_rt(m_total,'char_2_y_char_6_y_char_7_y_char_9_y_rt',['act_date_int_group_1_rt'])
# ######################################
# print('group_1 congregate') 
# ######################################
# m_total=congregate_small_category(m_total,'group_1',1)
######################################
print('final data treatment')
######################################
m_total['fill05']=0.5
m_total=final_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
df_train_gp=data_train[data_train['group_1'].isin(gp_intersect)].copy()
train_outcome_gp=df_train_gp['outcome']
df_train_wo=data_train.copy()
#df_train_wo=data_train[data_train['group_act_size']<=small_group_act_size].copy()
train_outcome_wo=df_train_wo['outcome']
######################################
print('columns for training/testing')
######################################
available_col=['activity_category', 'activity_id', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'outcome', 'people_id', 'char_1_y', 'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']
categorical_col=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_1_y', 'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y']
use_col_gp=['activity_category', 'char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x',  'group_1', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']
use_col_wo=['activity_category', 'char_1_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']#'char_10_x', 

available_col=available_col+['group_1_con', 'is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','is_weekend','act_week_num', 'act_date_int', 'peo_date_int', 'group_act_size','char_6_y_prod_char_2_y', 'char_5_y_prod_char_6_y','char_7_y_prod_char_9_y', 'char_1_y_prod_char_8_y','char_2_y_char_6_y_char_7_y_char_9_y_rt', 'group_1_rt','act_date_int_group_1_rt', 'act_date_int_group_1_rt_boundary','group_1_rt_s', 'group_1_rt_us', 'act_date_int_group_1_rt_boundary_s','act_date_int_group_1_rt_boundary_us','char_2_y_char_6_y_char_7_y_char_9_y_rt_s','char_2_y_char_6_y_char_7_y_char_9_y_rt_us', 'fill05']

categorical_col=categorical_col+['group_1_con','act_weekday','act_week_num', 'char_6_y_prod_char_2_y', 'char_5_y_prod_char_6_y','char_7_y_prod_char_9_y', 'char_1_y_prod_char_8_y']#'act_year', 'act_month', 'act_day',

use_col_gp=use_col_gp+['is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_week_num', 'is_weekend','act_date_int', 'peo_date_int', 'group_act_size','char_6_y_prod_char_2_y','char_2_y_char_6_y_char_7_y_char_9_y_rt', 'group_1_rt','act_date_int_group_1_rt', 'act_date_int_group_1_rt_boundary','group_1_rt_s', 'group_1_rt_us', 'act_date_int_group_1_rt_boundary_s','act_date_int_group_1_rt_boundary_us','char_2_y_char_6_y_char_7_y_char_9_y_rt_s','char_2_y_char_6_y_char_7_y_char_9_y_rt_us']

use_col_wo=use_col_wo+['group_1_con', 'is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','is_weekend', 'act_date_int', 'peo_date_int', 'char_5_y_prod_char_6_y','char_7_y_prod_char_9_y']#, 'char_1_y_prod_char_8_y']#'act_week_num',
######################################
print('df_train_gp water down')
######################################
df_train_gp.rename(columns={'act_date_int_group_1_rt':'act_date_int_group_1_rt_optimistic'},inplace=True)
df_train_gp.rename(columns={'group_1_rt':'group_1_rt_optimistic'},inplace=True)
df_train_gp.rename(columns={'act_date_int_group_1_rt_boundary':'act_date_int_group_1_rt_boundary_optimistic'},inplace=True)
df_train_gp.rename(columns={'char_2_y_char_6_y_char_7_y_char_9_y_rt':'char_2_y_char_6_y_char_7_y_char_9_y_rt_optimistic'},inplace=True)
del df_train_gp['group_1_rt_s']
del df_train_gp['group_1_rt_us']
del df_train_gp['act_date_int_group_1_rt_boundary_s']
del df_train_gp['act_date_int_group_1_rt_boundary_us']
del df_train_gp['char_2_y_char_6_y_char_7_y_char_9_y_rt_s']
del df_train_gp['char_2_y_char_6_y_char_7_y_char_9_y_rt_us']

mock_miss_act=df_train_gp.sample(frac=train_gp_frac).index
t_df_train_gp_frac=df_train_gp.loc[mock_miss_act]

df_train_gp=success_rate(df_train_gp,['act_date_int','group_1'],t_df_train_gp_frac)
df_train_gp=success_rate(df_train_gp,['char_2_y','char_6_y','char_7_y','char_9_y'],t_df_train_gp_frac)
df_train_gp=success_rate(df_train_gp,['group_1'],t_df_train_gp_frac)

t_df_train_gp_frac=df_train_gp.loc[mock_miss_act]
df_train_gp=interpolate_rt_df(df_train_gp,t_df_train_gp_frac,'group_1','act_date_int','act_date_int_group_1_rt','group_1_rt','group_act_size')
df_train_gp=interpolate_rt_df(df_train_gp,t_df_train_gp_frac,'group_1','act_date_int','act_date_int_group_1_rt','group_1_rt','group_act_size',True)

df_train_gp=secondary_feature(df_train_gp,'group_1_rt','fill05')
df_train_gp=secondary_feature(df_train_gp,'char_2_y_char_6_y_char_7_y_char_9_y_rt','fill05')

#We want to deal with the entries of the groups that are left out in the sampling. The way we do this is to assume their act_date_int_group_1_rt are all in the boundary, and are all 0.5
t_gp_intersect=np.intersect1d(df_train_gp['group_1'].values,t_df_train_gp_frac['group_1'].values)
t_act_date_int_group_1_rt=df_train_gp['act_date_int_group_1_rt']
t_act_date_int_group_1_rt[~df_train_gp['group_1'].isin(t_gp_intersect)]=0.5
df_train_gp['act_date_int_group_1_rt']=t_act_date_int_group_1_rt

df_train_gp=unmask_rt(df_train_gp,'group_1_rt',['act_date_int_group_1_rt'])
df_train_gp=unmask_rt(df_train_gp,'act_date_int_group_1_rt_boundary',['act_date_int_group_1_rt'])
df_train_gp=unmask_rt(df_train_gp,'char_2_y_char_6_y_char_7_y_char_9_y_rt',['act_date_int_group_1_rt'])

del t_df_train_gp_frac
del t_act_date_int_group_1_rt

df_train_gp=df_train_gp[use_col_gp]
######################################
print('gp one hot')
######################################
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
df_categorical_col_gp=list(set(use_col_gp)&set(categorical_col))
df_not_categorical_col_gp=list(set(use_col_gp)-set(categorical_col))
enc_gp=OneHotEncoder(handle_unknown='ignore')
enc_gp=enc_gp.fit(pd.concat([df_train_gp[df_categorical_col_gp],data_test[df_categorical_col_gp]]))
t_df_cat_train_gp=enc_gp.transform(df_train_gp[df_categorical_col_gp])
t_df_spr_train_gp=hstack((df_train_gp[df_not_categorical_col_gp],t_df_cat_train_gp))
dtrain_gp=xgb.DMatrix(t_df_spr_train_gp,label=train_outcome_gp)
del t_df_cat_train_gp
del t_df_spr_train_gp
######################################
print('prepare test_gp')
######################################
df_test_gp=data_test[data_test['group_1'].isin(gp_intersect)]
if not kaggle_output:
    df_test_gp=df_test_gp.merge(test_outcome,how='left',left_on='activity_id',right_on='activity_id')
    test_outcome_gp=df_test_gp['outcome_y']
df_test_gp=secondary_feature(df_test_gp,'char_2_y_char_6_y_char_7_y_char_9_y_rt','fill05')
df_test_gp=df_test_gp[use_col_gp]
t_df_cat_test_gp=enc_gp.transform(df_test_gp[df_categorical_col_gp])
t_df_spr_test_gp=hstack((df_test_gp[df_not_categorical_col_gp],t_df_cat_test_gp))
dtest_gp=xgb.DMatrix(t_df_spr_test_gp)
del t_df_cat_test_gp
del t_df_spr_test_gp
######################################
print('run test_gp')
######################################
watchlist  = [(dtrain_gp,'train')]
param = {'max_depth':10, 'eta':eta_gp, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst_gp = xgb.train(param, dtrain_gp, num_round_gp, watchlist,early_stopping_rounds=early_stopping_rounds)
######################################
print('df_train_wo water down')
######################################
# df_train_wo.rename(columns={'char_6_y_char_38_rt':'char_6_y_char_38_rt_optimistic'},inplace=True)
# df_train_wo.rename(columns={'char_5_y_char_7_y_char_32_rt':'char_5_y_char_7_y_char_32_rt_optimistic'},inplace=True)
# del df_train_wo['char_6_y_char_38_rt_s']
# del df_train_wo['char_6_y_char_38_rt_us']
# del df_train_wo['char_5_y_char_7_y_char_32_rt_s']
# del df_train_wo['char_5_y_char_7_y_char_32_rt_us']

# mock_miss_act=df_train_wo.sample(frac=train_wo_frac).index
# t_df_train_wo_frac=df_train_wo.loc[mock_miss_act]

# df_train_wo=success_rate(df_train_wo,['char_6_y','char_38'],t_df_train_wo_frac)
# df_train_wo=success_rate(df_train_wo,['char_5_y','char_7_y','char_32'],t_df_train_wo_frac)

# df_train_wo=unmask_rt(df_train_wo,'char_5_y_char_7_y_char_32_rt',['char_6_y_char_38_rt'])
# df_train_wo=unmask_rt(df_train_wo,'char_6_y_char_38_rt',['char_5_y_char_7_y_char_32_rt'])

# df_train_wo=secondary_feature(df_train_wo,'char_6_y_char_38_rt','fill05')
# df_train_wo=secondary_feature(df_train_wo,'char_5_y_char_7_y_char_32_rt','fill05')

# del t_df_train_wo_frac

df_train_wo=df_train_wo[use_col_wo]
######################################
print('wo one hot')
######################################
df_categorical_col_wo=list(set(use_col_wo)&set(categorical_col))
df_not_categorical_col_wo=list(set(use_col_wo)-set(categorical_col))
enc_wo=OneHotEncoder(handle_unknown='ignore')
enc_wo=enc_wo.fit(pd.concat([df_train_wo[df_categorical_col_wo],data_test[df_categorical_col_wo]]))
t_df_cat_train_wo=enc_wo.transform(df_train_wo[df_categorical_col_wo])
t_df_spr_train_wo=hstack((df_train_wo[df_not_categorical_col_wo],t_df_cat_train_wo))
dtrain_wo=xgb.DMatrix(t_df_spr_train_wo,label=train_outcome_wo)
del t_df_cat_train_wo
del t_df_spr_train_wo
######################################
print('prepare test_wo')
######################################
df_test_wo=data_test[~data_test['group_1'].isin(gp_intersect)]
if not kaggle_output:
    df_test_wo=df_test_wo.merge(test_outcome,how='left',left_on='activity_id',right_on='activity_id')
    test_outcome_wo=df_test_wo['outcome_y']
df_test_wo=df_test_wo[use_col_wo]
t_df_cat_test_wo=enc_wo.transform(df_test_wo[df_categorical_col_wo])
t_df_spr_test_wo=hstack((df_test_wo[df_not_categorical_col_wo],t_df_cat_test_wo))
dtest_wo=xgb.DMatrix(t_df_spr_test_wo)
del t_df_cat_test_wo
del t_df_spr_test_wo
######################################
print('run test_wo')
######################################
watchlist  = [(dtrain_wo,'train')]
param = {'max_depth':10, 'eta':eta_wo, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':tree_build_subsample,'colsample_bytree': col_sample_tree,'min_child_weight':0,'booster':'gbtree' }
bst_wo = xgb.train(param, dtrain_wo, num_round_wo, watchlist,early_stopping_rounds=early_stopping_rounds)
######################################
print('output')
######################################
from sklearn.metrics import roc_auc_score
if not kaggle_output:
    gp_per=dtest_gp.num_row()/(dtest_gp.num_row()+dtest_wo.num_row())
    wo_per=dtest_wo.num_row()/(dtest_gp.num_row()+dtest_wo.num_row())
    gp_per_k=429614/498687
    wo_per_k=69073/498687
    auc_11=roc_auc_score(test_outcome_gp,bst_gp.predict(dtest_gp))
    auc_22=roc_auc_score(test_outcome_wo,bst_wo.predict(dtest_wo))
    auc_12=1-np.sqrt((1-auc_11)/2)
    print(auc_11)
    print(auc_22)
    print(roc_auc_score(np.concatenate((test_outcome_gp,test_outcome_wo)),np.concatenate((bst_gp.predict(dtest_gp),prediction_mod_funct(bst_wo.predict(dtest_wo),alpha,confidence_wo)))))
    print('kaggle optimistic',auc_11*gp_per_k**2+2*np.sqrt(auc_11)*gp_per_k*wo_per_k+auc_22*wo_per_k**2)
    print('kaggle estimate',auc_11*gp_per_k**2+2*0.989*gp_per_k*wo_per_k+auc_22*wo_per_k**2)
    print('kaggle lower_bound',auc_11*gp_per_k**2+2*auc_12*gp_per_k*wo_per_k+auc_22*wo_per_k**2)
    print('optimistic',auc_11*gp_per**2+2*np.sqrt(auc_11)*gp_per*wo_per+auc_22*wo_per**2)
    print('estimate',auc_11*gp_per**2+2*0.989*gp_per*wo_per+auc_22*wo_per**2)
    print('lower_bound',auc_11*gp_per**2+2*auc_12*gp_per*wo_per+auc_22*wo_per**2)
else:
    pred_gp=bst_gp.predict(dtest_gp)
    pred_wo=prediction_mod_funct(bst_wo.predict(dtest_wo),alpha,confidence_wo)
    act_id_gp=data_test[data_test['group_1'].isin(gp_intersect)]['activity_id']
    act_id_wo=data_test[~ data_test['group_1'].isin(gp_intersect)]['activity_id']
    output_gp = pd.DataFrame({ 'activity_id' : act_id_gp, 'outcome': pred_gp })
    output_wo = pd.DataFrame({ 'activity_id' : act_id_wo, 'outcome': pred_wo })
    output=output_gp.append(output_wo)
    output.to_csv('redhat_pred.csv', index = False)
