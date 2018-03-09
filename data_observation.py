######################################
print('parameters')
######################################
fillna_num=60000#fill number cannot be negative, cannot be too large
sample_frac=1
small_group_act_size=150
######################################
print('functions')
######################################
def interpolate_funct(x,x_0,x_1,y_0,y_1,center=0.5,alpha=0.5):#alpha does not change gp_AUC
    l=2*(x-x_0)/(x_1-x_0)-1 #l(x_1)=1,l(x_0)=-1
    phi=np.tanh(alpha*l)/np.tanh(alpha)#phi(1)=1,phi(-1)=-1
    return (y_1-y_0)*phi/2+(y_1+y_0)/2

def interpolate_funct_boundary(x,x_0,y_0,boundary,decay,alpha=0.00001,alpha_limit=1):
#    boundary_limit=(y_0+decay*alpha_limit*boundary)/(1+decay*alpha_limit*boundary)
    boundary_limit=0
    return (y_0-boundary_limit)*np.exp(-alpha*decay*abs(x-x_0))+boundary_limit

def interpolate_rt(df_sorted_by_date,col_of_date_int,col_of_average,col_of_boundary,col_of_decay,date_int):
    if date_int > df_sorted_by_date[col_of_date_int].iloc[-1]:
        return interpolate_funct_boundary(date_int,df_sorted_by_date[col_of_date_int].iloc[-1],df_sorted_by_date[col_of_average].iloc[-1],df_sorted_by_date[col_of_boundary].iloc[-1],df_sorted_by_date[col_of_decay].iloc[-1])
    if date_int < df_sorted_by_date[col_of_date_int].iloc[0]:
        return interpolate_funct_boundary(date_int,df_sorted_by_date[col_of_date_int].iloc[0],df_sorted_by_date[col_of_average].iloc[0],df_sorted_by_date[col_of_boundary].iloc[0],df_sorted_by_date[col_of_decay].iloc[0])
    ind=df_sorted_by_date[col_of_date_int].searchsorted(date_int)[0]
    if date_int==df_sorted_by_date[col_of_date_int].iloc[ind]:
        return df_sorted_by_date[col_of_average].iloc[ind]
    return interpolate_funct(date_int,df_sorted_by_date[col_of_date_int].iloc[ind-1],df_sorted_by_date[col_of_date_int].iloc[ind],df_sorted_by_date[col_of_average].iloc[ind-1],df_sorted_by_date[col_of_average].iloc[ind])

def interpolate_rt_wrapped(df_to_interpolate,gpb_sorted,col_of_date_int,col_of_average,col_of_boundary,col_of_decay):
    rt=df_to_interpolate.apply(lambda entry : interpolate_rt(gpb_sorted.loc[entry[0]],col_of_date_int,col_of_average,col_of_boundary,col_of_decay,entry[1]),axis=1)
    return rt

def interpolate_rt_df(df_to_interpolate,df_data,group_by,col_of_date_int,col_of_average,col_of_boundary,col_of_decay):
#df_to_interpolate needs to have two cols, with group_by being the fisrt and col_of_date_int the second
#the function will destroy df_to_interpolate
    gpb=df_data.groupby(group_by)
    gpb_sorted=gpb[[col_of_date_int,col_of_average,col_of_boundary,col_of_decay]].apply(partial(pd.DataFrame.sort_values,by=col_of_date_int))
    df_to_interpolate.drop_duplicates(inplace=True)
    with Pool(6) as p:
        df_to_interpolate_split=np.array_split(df_to_interpolate,6)
        df_to_interpolate[col_of_average]=pd.concat(p.map(partial(interpolate_rt_wrapped,gpb_sorted=gpb_sorted,col_of_date_int=col_of_date_int,col_of_average=col_of_average,col_of_boundary=col_of_boundary,col_of_decay=col_of_decay),df_to_interpolate_split))
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
act_train=pd.read_csv('../act_train.csv',parse_dates=['date'])
peo_data=pd.read_csv('../people.csv',parse_dates=['date'])
######################################
print('train test split')
######################################
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

# #############################################
# print('digression to group_act_size')
# #############################################
# gpb=m_total.groupby('group_1')
# t_group_act_size=gpb['activity_id'].count()
# t_group_act_size.name='group_act_size'
# m_total=m_total.merge(pd.DataFrame(t_group_act_size),how='left',left_on='group_1',right_index=True)
# #############################################
# print('batch feature engineer')
# #############################################

# available_features_prod=['char_1_y', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_38', 'char_10_x']

# available_features=  ['char_1_y', 'char_2_y','char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y','char_9_y', 'char_10_y', 'char_11', 'char_12', 'char_13', 'char_14','char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20','char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26','char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32','char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38','char_1_x', 'char_10_x', 'char_2_x','char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x','char_9_x']#'is_last_act', 'act_year', 'act_month', 'act_day', 'act_weekday','act_week_num','group_1'

# m_train=m_total.iloc[:len(act_train)]
# m_small=m_train[m_train['group_act_size']<=small_group_act_size]
# for col1 in available_features_prod:
#     for col2 in available_features_prod:
#         if col1<col2:
#             print(col1,'  ',col2)
#             m_total=success_rate(m_total,[col1,col2],m_small)

# m_train=m_total.iloc[:len(act_train)]
# m_small=m_train[m_train['group_act_size']<=small_group_act_size]
# for col2 in available_features:
#     print(col2)
#     m_total=success_rate(m_total,['char_5_y','char_7_y',col2],m_small)

# m_train=m_total.iloc[:len(act_train)]
# m_small=m_train[m_train['group_act_size']<=small_group_act_size]
# m_total=success_rate(m_total,['char_5_y','char_7_y','char_32'],m_small)

#############################################
print('group date interpolation')
#############################################
from functools import partial
m_train=m_total.iloc[:len(act_train)]
m_test=m_total.iloc[len(act_train):]
m_test_gp=m_test[m_test['group_1'].isin(gp_intersect)]
m_test_gp=m_test_gp[['group_1','act_date_int']]
m_test_gp=interpolate_rt_df(m_test_gp,m_train,'group_1','act_date_int','act_date_int_group_1_rt','group_1_rt','group_act_size')
del m_test['act_date_int_group_1_rt']
m_test=m_test.merge(m_test_gp,how='left',on=['group_1','act_date_int'])
m_test['act_date_int_group_1_rt'].fillna(fillna_num,inplace=True)
del m_total['act_date_int_group_1_rt']
m_total['act_date_int_group_1_rt']=pd.concat([m_train['act_date_int_group_1_rt'],m_test['act_date_int_group_1_rt']]).reset_index(drop=True)
del m_test
del m_test_gp
######################################
print('final data treatment')
######################################
m_total['fill05']=0.5
m_total=final_data_treat(m_total)
data_train=m_total.iloc[:len(act_train)]
data_test=m_total.iloc[len(act_train):]
######################################
print('prepare test group')
######################################
df_test_gp=data_test[data_test['group_1'].isin(gp_intersect)]
df_test_wo=data_test[~data_test['group_1'].isin(gp_intersect)]
df_test_gp=df_test_gp.merge(local_outcome,how='left',left_on='activity_id',right_on='activity_id')
df_test_wo=df_test_wo.merge(local_outcome,how='left',left_on='activity_id',right_on='activity_id')
local_outcome_gp=df_test_gp['outcome_y']
local_outcome_wo=df_test_wo['outcome_y']
######################################
print('output')
######################################
from sklearn.metrics import roc_auc_score

for col in df_test_gp.columns:
    if df_test_gp[col].dtype=='float64':
        df_test_gp=secondary_feature(df_test_gp,col,'fill05')
        print(col,' AUC: ',roc_auc_score(local_outcome_gp,df_test_gp[col]))

# for col in df_test_wo.columns:
#     if df_test_wo[col].dtype=='float64':
#         df_test_gp=secondary_feature(df_test_wo,col,'fill05')
#         print(col,' AUC: ',roc_auc_score(local_outcome_wo,df_test_wo[col]))
###################################

#        print(col,' AUC: ',roc_auc_score(local_outcome_gp,df_test_gp[col]))
#act_dat_int_group_1_rt AUC: ~0.9967
#char_2_y_char_6_y_rt  AUC:  0.9265657772037466
#char_2_y_char_7_y_rt  AUC:  0.9435913817847027
#char_2_y_char_6_y_char_9_y_rt  AUC:  0.9402275075309572
#char_2_y_char_6_y_char_7_y_rt  AUC:  0.9454846825478471
#char_2_y_char_6_y_char_38_rt  AUC:  0.946604653620611
#char_2_y_char_6_y_char_9_y_char_38_rt  AUC:  0.9442989413847488
#char_2_y_char_6_y_char_9_y_char_7_y_rt  AUC:  0.9477917021151983
#char_2_y_char_6_y_char_9_y_char_7_y_group_1_rt  AUC:  0.9598051120424427
#group_1_rt  AUC:  0.9913754411819684
#
#small group is very useful
#the optimal size is around 150


#        print(col,' AUC: ',roc_auc_score(local_outcome_wo,df_test_wo[col]))
#char_6_y_char_3_y_rt  AUC:  0.6282638903826605
#char_6_y_char_4_y_rt  AUC:  0.6251580544331923
#char_6_y_char_5_y_rt  AUC:  0.6300210100223461
# char_7_y_char_1_y_rt  AUC:  0.6137280005293472
# char_7_y_char_2_y_rt  AUC:  0.6137280005293472
# char_7_y_char_3_y_rt  AUC:  0.6259568603131355
# char_7_y_char_4_y_rt  AUC:  0.6167377707582318
# char_7_y_char_5_y_rt  AUC:  0.6324811479210652
# char_7_y_char_6_y_rt  AUC:  0.6235398284007544
# char_7_y_char_38_rt  AUC:  0.6154149483198303
# char_8_y_char_6_y_rt  AUC:  0.6186663646167767
# char_8_y_char_7_y_rt  AUC:  0.6333556973073855
# char_9_y_char_6_y_rt  AUC:  0.6176003189088146
# char_9_y_char_7_y_rt  AUC:  0.6378370233711562

#char_5_y_char_6_y_char_18_rt  AUC:  0.6346309400054274
#char_7_y_char_9_y_char_32_rt  AUC:  0.6439809098167414

# char_1_y_char_2_y_rt  AUC:  0.5273437304999757
# char_1_y_char_3_y_rt  AUC:  0.5496091310459286
# char_1_y_char_4_y_rt  AUC:  0.5300337072444317
# char_1_y_char_5_y_rt  AUC:  0.545174871441463
# char_1_y_char_6_y_rt  AUC:  0.5699339346141727
# char_1_y_char_7_y_rt  AUC:  0.6547890537776684
# char_1_y_char_8_y_rt  AUC:  0.5747431521417448
# char_1_y_char_9_y_rt  AUC:  0.5900944872822356
# char_1_y_char_38_rt  AUC:  0.5741896476878694
# char_2_y_char_3_y_rt  AUC:  0.5496091310459286
# char_2_y_char_4_y_rt  AUC:  0.5300337072444317
# char_2_y_char_5_y_rt  AUC:  0.545174871441463
# char_2_y_char_6_y_rt  AUC:  0.5699339346141727
# char_2_y_char_7_y_rt  AUC:  0.6547890537776684
# char_2_y_char_8_y_rt  AUC:  0.5747431521417448
# char_2_y_char_9_y_rt  AUC:  0.5900944872822356
# char_2_y_char_38_rt  AUC:  0.5741896476878694
# char_3_y_char_4_y_rt  AUC:  0.5267325566160793
# char_3_y_char_5_y_rt  AUC:  0.5468432191528538
# char_3_y_char_6_y_rt  AUC:  0.6002909357244526
# char_3_y_char_7_y_rt  AUC:  0.6463892262270035
# char_3_y_char_8_y_rt  AUC:  0.573546113952254
# char_3_y_char_9_y_rt  AUC:  0.5765912067313828
# char_4_y_char_5_y_rt  AUC:  0.5247023729027899
# char_4_y_char_6_y_rt  AUC:  0.5822082046189432
# char_4_y_char_7_y_rt  AUC:  0.6452651370685072
# char_4_y_char_8_y_rt  AUC:  0.5678456469418435
# char_4_y_char_9_y_rt  AUC:  0.5840018730648614
# char_5_y_char_6_y_rt  AUC:  0.599347292090088
# char_5_y_char_7_y_rt  AUC:  0.6544299558965125
# char_5_y_char_8_y_rt  AUC:  0.5769802101502912
# char_5_y_char_9_y_rt  AUC:  0.5845463256325576
# char_6_y_char_7_y_rt  AUC:  0.658468618251189
# char_6_y_char_8_y_rt  AUC:  0.612540712295831
# char_6_y_char_9_y_rt  AUC:  0.6229576540172288
# char_7_y_char_8_y_rt  AUC:  0.6706338123861303
# char_7_y_char_9_y_rt  AUC:  0.6805899689532917
# char_8_y_char_9_y_rt  AUC:  0.5749434780365847
# char_38_char_3_y_rt  AUC:  0.5640137696249259
# char_38_char_4_y_rt  AUC:  0.5622080070552206
# char_38_char_5_y_rt  AUC:  0.5634828232667091
# char_38_char_6_y_rt  AUC:  0.6043722818699975
# char_38_char_7_y_rt  AUC:  0.6604434852979764
# char_38_char_8_y_rt  AUC:  0.5841081632455807
# char_38_char_9_y_rt  AUC:  0.5815817575477856
# char_10_x_char_1_y_rt  AUC:  0.5274437063807825
# char_10_x_char_2_y_rt  AUC:  0.5274437063807825
# char_10_x_char_3_y_rt  AUC:  0.5174832558331319
# char_10_x_char_4_y_rt  AUC:  0.529006678399955
# char_10_x_char_5_y_rt  AUC:  0.5256565312842497
# char_10_x_char_6_y_rt  AUC:  0.5594235905327593
# char_10_x_char_7_y_rt  AUC:  0.6036501993979375
# char_10_x_char_8_y_rt  AUC:  0.5377067810024649
# char_10_x_char_9_y_rt  AUC:  0.5454013567315056
# char_10_x_char_38_rt  AUC:  0.5228249648891466
#will use char_7_y_char_9_y,char_5_y_char_6_y,char_1_y_char_8_y
