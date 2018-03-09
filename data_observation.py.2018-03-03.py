#do not take NA
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

def feature_importance(data,related_features):
#use only on categorical features
    file_name=open('feature_importance','a')
    important_features=[]
    for col in related_features:
        datapg=data.groupby(by=col)
        datapg_count=datapg['outcome'].count()
        datapg_outcome=datapg['outcome'].mean()
        datapg_outcome=datapg_outcome[datapg_count>=10]
        file_name.write(col+':  '+'\n')
        print(col+':  ')
        file_name.write(str((0.5-np.abs(datapg_outcome-0.5)).mean())+'\n')
        print((0.5-np.abs(datapg_outcome-0.5)).mean())
#        file_name.write(str(len(np.unique(data[col].values)))+'\n')
#        print(len(np.unique(data[col].values)))
        if (0.5-np.abs(datapg_outcome-0.5)).mean()<0.25:
            important_features.append(col)

    unimportant_features=list(set(related_features)-set(important_features))
    for col1 in ['super_prod']:#:related_features
        file_name.write('col1 is now: '+col1+'\n')
        print('col1 is now: '+col1)
        for col2 in unimportant_features:
            if col1!=col2:
                datapg=data.groupby([col1,col2])
                datapg_count=datapg['outcome'].count()
                datapg_outcome=datapg['outcome'].mean()
                datapg_outcome=datapg_outcome[datapg_count>=5]
                if (0.5-np.abs(datapg_outcome-0.5)).mean()< 0.25:
                    file_name.write(col1+' '+col2+':  '+'\n')
                    print(col1+' '+col2+':  ')
                    file_name.write(str((0.5-np.abs(datapg_outcome-0.5)).mean())+'\n')
                    print((0.5-np.abs(datapg_outcome-0.5)).mean())
                    file_name.write(str(datapg_count.max())+'\n')
                    print(datapg_count.max())
                    file_name.write(str(datapg_count.mean())+'\n')
                    print(datapg_count.mean())
    file_name.close()




gp=data_train.groupby('group_1')
gp_num_act=gp['activity_id'].count()
gp_num_act.name='gp_num_act'
data_train=data_train.merge(pd.DataFrame(gp_num_act),how='left',left_on='group_1',right_index=True)
small_data=data_train[data_train['gp_num_act']<5000]

small_data=prod_feature(small_data,'char_6_y','char_7_y','super_rt')
small_data=prod_feature(small_data,'super_prod','char_32','super_rt')
small_data=prod_feature(small_data,'super_prod','char_18','super_rt')
small_data=prod_feature(small_data,'super_prod','char_25','super_rt')
small_data=prod_feature(small_data,'super_prod','char_23','super_rt')
small_data=prod_feature(small_data,'super_prod','char_29','super_rt')

feature_importance(small_data,related_col)


for col in small_data.columns:
    print(col)
    print(len(np.unique(small_data[col].values)))

related_col=['activity_category',
 'char_1_y',
 'char_2_y',
 'char_3_y',
 'char_4_y',
 'char_5_y',
 'char_6_y',
 'char_7_y',
 'char_8_y',
 'char_9_y',
 'char_10_y',
 'char_11',
 'char_12',
 'char_13',
 'char_14',
 'char_15',
 'char_16',
 'char_17',
 'char_18',
 'char_19',
 'char_20',
 'char_21',
 'char_22',
 'char_23',
 'char_24',
 'char_25',
 'char_26',
 'char_27',
 'char_28',
 'char_29',
 'char_30',
 'char_31',
 'char_32',
 'char_33',
 'char_34',
 'char_35',
 'char_36',
 'char_37']


char_7_y char_6_y:  
0.1980179864905781
super_prod char_32:  
0.18835317304567042
super_prod char_18:  
0.177214369437007
super_prod char_25:  
0.168586332992773
super_prod char_23:  
0.16002527557397178
super_prod char_29:  
0.14564482150730015
