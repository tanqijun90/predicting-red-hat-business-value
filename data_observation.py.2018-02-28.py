def prod_features(col1,unimportant_features):
    for col2 in unimportant_features:
        if col1!=col2:
            datapg=data.groupby([col1,col2])
            datapg_count=datapg['outcome'].count()
            datapg_outcome=datapg['outcome'].mean()
            datapg_outcome=datapg_outcome[datapg_count>=10]
            if (0.5-np.abs(datapg_outcome-0.5)).mean()< 0.2:
                print(col1+' '+col2+':  ')
                print((0.5-np.abs(datapg_outcome-0.5)).mean())

def feature_importance(data,related_features):
#use only on categorical features
    important_features=[]
    for col in related_features:
        datapg=data.groupby(by=col)
        datapg_count=datapg['outcome'].count()
        datapg_outcome=datapg['outcome'].mean()
        datapg_outcome=datapg_outcome[datapg_count>=10]
        print(col+':  ')
        print((0.5-np.abs(datapg_outcome-0.5)).mean())
        if (0.5-np.abs(datapg_outcome-0.5)).mean()<0.3:
            important_features.append(col)

    unimportant_features=list(set(related_features)-set(important_features))
    from multiprocessing import Pool
    from functoools import parital
    with Pool(6) as p:
        p.map(parital(prod_features,unimportant_features=unimportant_features),unimportant_features)

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


act_data=pd.read_csv('act_train.csv',parse_dates=['date'])
peo_data=pd.read_csv('people.csv',parse_dates=['date'])
act_test=pd.read_csv('act_test.csv',parse_dates=['date'])
tsdata=act_test.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
trdata=act_data.merge(peo_data,how='left',left_on='people_id',right_on='people_id')
print(trdata.columns)
related_features=['people_id','date_x', 'activity_category', 'char_1_x',
       'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x',
       'char_8_x', 'char_9_x', 'char_10_x', 'char_1_y', 'group_1',
       'char_2_y', 'date_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y',
       'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
       'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18',
       'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24',
       'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30',
       'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
            'char_37', 'char_38']

#Each activity either have char_10 or have char_1 to char_9

##########################################
# The following tells one how good one
# feature is in predicting the label
##########################################
###########################################

tsdata['group_1'].value_counts()
trdata['group_1'].value_counts()
print(len(np.unique(tsdata['group_1'].values)))
print(len(np.unique(trdata['group_1'].values)))
print(len(np.unique((trdata['group_1'].append(tsdata['group_1'])).values)))

trgp=trdata.groupby(by='group_1')
trgp_count=trgp['outcome'].count()
trgp_outcome=trgp['outcome'].mean()
trgp_outcome=trgp_outcome[trgp_count>=10]
len(trgp_outcome)
len(trgp_count)
print((0.5-np.abs(trgp_outcome-0.5)).mean())
#Therefore gp average is a very good predictor
#The problem is that how could one deal with groups that are not in?
#Trees cannot discover those hidden second layers. We need to create product features to help it with that.
#

tr10=trdata.groupby(by='char_10_x')
tr10_count=tr10['char_10_x'].count()
tr10_outcome=tr10['outcome'].mean()
tr10_outcome=tr10_outcome[tr10_count>=100]
len(tr10_outcome)
len(tr10_count)
print((0.5-np.abs(tr10_outcome-0.5)).mean())
#char_10 is a weak predictor, but still useful.
