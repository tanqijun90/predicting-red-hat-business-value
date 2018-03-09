act_data=pd.read_csv('act_train.csv',parse_dates=['date'])
peo_data=pd.read_csv('people.csv',parse_dates=['date'])
act_test=pd.read_csv('act_test.csv',parse_dates=['date'])

#act_data.drop(columns='activity_id',inplace=True)
#act_test.drop(columns='activity_id',inplace=True)

peo_data.columns=['people_id', 'pchar_1', 'group_1', 'pchar_2', 'date', 'pchar_3', 'pchar_4','pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9', 'pchar_10', 'pchar_11','pchar_12', 'pchar_13', 'pchar_14', 'pchar_15', 'pchar_16', 'pchar_17','pchar_18', 'pchar_19', 'pchar_20', 'pchar_21', 'pchar_22', 'pchar_23','pchar_24', 'pchar_25', 'pchar_26', 'pchar_27', 'pchar_28', 'pchar_29','pchar_30', 'pchar_31', 'pchar_32', 'pchar_33', 'pchar_34', 'pchar_35','pchar_36', 'pchar_37', 'pchar_38']

act_data.replace(to_replace='type ',regex=True,value='',inplace=True)
act_test.replace(to_replace='type ',regex=True,value='',inplace=True)
peo_data.replace(to_replace='type ',regex=True,value='',inplace=True)
peo_data['group_1'].replace(to_replace='group ',regex=True,value='',inplace=True)

act_data['month']=act_data['date'].dt.month
act_data['day']=act_data['date'].dt.day
act_data['weekday']=act_data['date'].dt.weekday
act_data['date']=(act_data['date']-pd.datetime(2020,1,1))

act_test['month']=act_test['date'].dt.month
act_test['day']=act_test['date'].dt.day
act_test['weekday']=act_test['date'].dt.weekday
act_test['date']=(act_test['date']-pd.datetime(2020,1,1))

peo_data['date']=(peo_data['date']-pd.datetime(2020,1,1))

act_data['date']=act_data['date'].apply(lambda x: x.days)
act_test['date']=act_test['date'].apply(lambda x: x.days)
peo_data['date']=peo_data['date'].apply(lambda x: x.days)

act_data.fillna(-1,inplace=True)
act_test.fillna(-1,inplace=True)

for col in ['activity_category', 'char_1', 'char_2', 'char_3','char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10']:
    act_data[col]=pd.to_numeric(act_data[col],errors='coerce')
    act_data[col]=act_data[col].astype('int32',errors='ignore')
    act_test[col]=pd.to_numeric(act_test[col],errors='coerce')
    act_test[col]=act_test[col].astype('int32',errors='ignore')
for col in ['pchar_1', 'group_1', 'pchar_2', 'date', 'pchar_3','pchar_4', 'pchar_5', 'pchar_6', 'pchar_7', 'pchar_8', 'pchar_9']:
    peo_data[col]=pd.to_numeric(peo_data[col],errors='coerce')
    peo_data[col]=peo_data[col].astype('int32',errors='ignore')
for col in ['pchar_10', 'pchar_11', 'pchar_12', 'pchar_13', 'pchar_14', 'pchar_15','pchar_16', 'pchar_17','pchar_18', 'pchar_19', 'pchar_20', 'pchar_21','pchar_22', 'pchar_23', 'pchar_24', 'pchar_25', 'pchar_26', 'pchar_27','pchar_28', 'pchar_29', 'pchar_30', 'pchar_31', 'pchar_32', 'pchar_33','pchar_34', 'pchar_35', 'pchar_36', 'pchar_37']:
    peo_data[col]=peo_data[col].astype('int8')

for col in act_data.columns[1:]:
    print('{}{}'.format(col+'  ',len(act_data[col].unique())))
for col in act_test.columns[1:]:
    print('{}{}'.format(col+'  ',len(act_test[col].unique())))
for col in peo_data.columns[1:]:
    print('{}{}'.format(col+'  ',len(peo_data[col].unique())))
