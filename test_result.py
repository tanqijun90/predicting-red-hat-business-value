
#######################################
#In [7]: data_test['act_week_num_group_rt_s'].mean()
#Out[7]: 0.2971302640734569

#In [8]: data_test['act_week_num_group_rt_us'].mean()
#Out[8]: 0.3758489794199568

#In [9]: data_test['group_rt_s'].mean()
#Out[9]: 0.4447479080064249

#In [10]: data_test['group_rt_us'].mean()
#Out[10]: 0.4167423654516761

#In [13]: data_test['char_10_x_char_38_rt_s'].mean()
#Out[13]: 0.5149783331027278

#In [14]: data_test['char_10_x_char_38_rt_us'].mean()
#Out[14]: 0.3131523380396922
###########################################
# m_test_small['act_date_int_group_1_rt'].value_counts()
#Out[155]: 
#0.000000        80063
#1.000000        58175
#60000.000000    20844
#len(m_test_small)
#Out[156]: 160173

20844/160173
#.13013429229645445861
###########################################
len(m_test)
#Out[32]: 498687
 len(m_test_wo)
#Out[33]: 69073
m_test_wo['group_1'].value_counts()
#Out[34]: 
# group 20876    868
# group 33904    490
# group 26933    416
# group 2665     397
# group 28885    388
# group 26426    383
# group 18704    322
# group 25073    318
# group 8314     296
# group 6895     283
# group 10206    276
# group 21802    275
# group 34084    270
# group 1805     259
# group 34251    252
# group 9340     251
# group 18       242
# group 28727    239
# group 23112    234
# group 18831    224
# group 33180    222
# group 46339    220
# group 18476    217
# group 9816     210
# group 35802    208
# group 38735    202
# group 21659    202
# group 18152    198
# group 27790    194
# group 25804    192

m_test['group_1'].value_counts()
# Out[35]: 
# group 17304    165604
# group 667        3904
# group 8386       2194
# group 17899      1224
# group 9280       1166
# group 450         989
# group 1270        911
# group 3598        889
# group 1482        883
# group 20876       868
# group 18559       853
# group 15723       833
# group 418         784
# group 7256        769
# group 9702        754
# group 1490        750
# group 5149        729
# group 3702        711
# group 7124        693
# group 3229        686
# group 142         671
# group 11143       641
# group 7403        614
# group 956         602
# group 9439        576
# group 28670       568
# group 15588       549
# group 1277        532
# group 33904       490
# group 752         483

m_test_wo['group_1'].value_counts().describe()
# Out[38]: 
# count    4325.000000
# mean       15.970636
# std        33.888930
# min         1.000000
# 25%         2.000000
# 50%         6.000000
# 75%        16.000000
# max       868.000000

#a small group is one whose size smaller than 100
################################################
print(col,' AUC: ',roc_auc_score(local_outcome_gp,df_test_gp[col]))
#char_2_y_char_6_y_rt  AUC:  0.9265657772037466
#char_2_y_char_7_y_rt  AUC:  0.9435913817847027
#char_2_y_char_6_y_char_9_y_rt  AUC:  0.9402275075309572
#char_2_y_char_6_y_char_7_y_rt  AUC:  0.9454846825478471
#char_2_y_char_6_y_char_38_rt  AUC:  0.946604653620611
#char_2_y_char_6_y_char_9_y_char_38_rt  AUC:  0.9442989413847488
#char_2_y_char_6_y_char_9_y_char_7_y_rt  AUC:  0.9477917021151983
#char_2_y_char_6_y_char_9_y_char_7_y_group_1_rt  AUC:  0.9598051120424427


