def hist_bin(pds,bin):
    hist_b=pds.value_counts()
    hist=((hist_b.cumsum()-hist_b/2)/hist_b.sum()*bin).round().astype('int32')
    print('{}{}'.format('Actual bin number: ',len(np.unique(hist.values))))
    return hist

bin=100
red_act=['char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10']
red_peo=['group_1','pchar_3','pchar_4']

#should use multiprocessing+pool.map
act_total=act_data.append(act_test,ignore_index=True)
for col in red_act:
    red_ind=hist_bin(act_total[col],100)
    act_data[col+'_red']=act_data[col].apply(lambda x: red_ind.loc[x])
    act_test[col+'_red']=act_test[col].apply(lambda x: red_ind.loc[x])

for col in red_peo:
    red_ind=hist_bin(peo_data[col],100)
    peo_data[col+'_red']=peo_data[col].apply(lambda x: red_ind.loc[x])

##############################################

use_act=act_data[['people_id', 'date', 'activity_category','outcome', 'month', 'day', 'weekday', 'char_1_red', 'char_2_red','char_3_red', 'char_4_red', 'char_5_red', 'char_6_red', 'char_7_red','char_8_red', 'char_9_red', 'char_10_red']].copy()
use_test=act_test[['people_id', 'date', 'activity_category','month', 'day', 'weekday', 'char_1_red', 'char_2_red','char_3_red', 'char_4_red', 'char_5_red', 'char_6_red', 'char_7_red','char_8_red', 'char_9_red', 'char_10_red']].copy()
use_peo=peo_data[['people_id', 'pchar_1','pchar_2', 'date','pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9',
'pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15','pchar_16', 'pchar_17', 'pchar_18', 'pchar_19', 'pchar_20', 'pchar_21','pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27',
'pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33','pchar_34', 'pchar_35', 'pchar_36', 'pchar_37', 'pchar_38','group_1_red', 'pchar_3_red', 'pchar_4_red']].copy()

df=pd.merge(use_act,use_peo,how='left',left_on='people_id',right_on='people_id')
df.drop(columns=['people_id'],inplace=True)
test_df=pd.merge(use_test,use_peo,how='left',left_on='people_id',right_on='people_id')
test_df.drop(columns=['people_id'],inplace=True)

not_categorical=['date_x', 'month', 'day', 'weekday', 'date_y', 'pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15', 'pchar_16', 'pchar_17', 'pchar_18', 'pchar_19', 'pchar_20', 'pchar_21', 'pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27', 'pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33', 'pchar_34', 'pchar_35', 'pchar_36', 'pchar_37', 'pchar_38','outcome']

categorical=[]
for category in df.columns:
    if category not in not_categorical:
        categorical.append(category)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc=enc.fit(pd.concat([df[categorical],test_df[categorical]]))

#####################################################

df_sam=df.sample(frac=0.2)
df_train,df_test=train_test_split(df_sam,test_size=0.5)
res_train=df_train['outcome']
res_test=df_test['outcome']

df_cat_train=enc.transform(df_train[categorical])
df_cat_test=enc.transform(df_test[categorical])
test_df_cat=enc.transform(test_df[categorical])

from scipy.sparse import hstack
df_spr_train=hstack((df_train[not_categorical[:-1]],df_cat_train))
df_spr_test=hstack((df_test[not_categorical[:-1]],df_cat_test))
test_df_spr=hstack((test_df[not_categorical[:-1]],test_df_cat))
dtrain=xgb.DMatrix(df_spr_train,label=res_train)
dtest=xgb.DMatrix(df_spr_test)
test_dtest=xgb.DMatrix(test_df_spr)

#####################################################

early_stopping_rounds=100
watchlist  = [(dtrain,'train')]
from sklearn.metrics import roc_auc_score

param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic','nthread':6,'eval_metric':'auc','subsample':0.7,'colsample_bytree': 0.7,'min_child_weight':0,'booster':'gbtree' }
num_round = 5000
bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)

roc_auc_score(res_test,bst.predict(dtest))

test_pred=bst.predict(test_dtest)
act_test2=pd.read_csv('act_test.csv',parse_dates=['date'])
output = pd.DataFrame({ 'activity_id' : act_test2['activity_id'], 'outcome': test_pred })
output.to_csv('redhat_pred.csv', index = False)

###########################################
colsam=[0.3,0.7]
nround=[1000,3000,5000,7000]
perf=np.zeros((len(colsam),len(nround)))
for i in range(len(colsam)):
    for j in range(len(nround)):
        param['colsample_bytree']= colsam[i]
        num_round=nround[j]
        bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)
        perf[i,j]=roc_auc_score(res_test,bst.predict(dtest))
        perf
