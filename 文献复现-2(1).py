# #%%
# '''
# 对于正向指标我们采用从小到大排，对于逆向指标我们采用从大到小排，例如： IVOL 越低越好，为逆向指标，采用倒序排列。
# '''
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')
'''
处理日频数据:
月个股交易股数 ：月内该股票的交易数量之和。
'''
data = pd.read_csv(r"C:\Users\zeng\Desktop\数据\股票日频行情信息表(后复权).csv").iloc[:,1:].sort_values(['月收盘日期','证券代码'])
data = data[(data['月收盘日期']<='2014-12-31') & (data['月收盘日期']>='2004-11-30')]
# 获取此次实证所需要的股票代码
month_data = pd.read_csv('财务数据整合好的月频数据.csv').replace([np.inf, -np.inf], np.nan).drop(columns=['stock_monthly_ret'])
stock_list = month_data['ts_code'].unique().tolist()

filtered_data = data[data['证券代码'].isin(stock_list)].reset_index().drop(columns='index')
filtered_data['月收盘日期'] = pd.to_datetime(filtered_data['月收盘日期'],format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)
filtered_data['流通股数'] = filtered_data['月个股流通市值']/filtered_data['月收盘价']

# 把收益率转为百分比形式
filtered_data['日收益率'] = filtered_data.groupby('证券代码')['月收盘价'].pct_change()*100

# 1. 计算个股流通市值对数
filtered_data['流通市值对数'] = np.log(filtered_data['月个股流通市值'])
# 2. 计算日Amihud
filtered_data['日Amihud'] = abs(filtered_data['日收益率'])/filtered_data['月个股交易金额']
Aminhud_df = filtered_data.groupby(['证券代码','交易月份'])['日Amihud'].mean().reset_index().rename(columns={'日Amihud':'月Amihud'})
# 3. 计算换手率
filtered_data['日换手率'] = filtered_data['月个股交易股数']/filtered_data['流通股数']
time_list = month_data['trade_date'].unique().tolist() + [20041231]
filtered_data = filtered_data[filtered_data['月收盘日期'].isin(time_list)].reset_index().drop(columns=['index','日Amihud'])
merge_df = pd.merge(filtered_data, Aminhud_df,on=['证券代码','交易月份'],how='left')
# 4. 计算月收益率,收益率转为百分比形式
merge_df['月收益率'] = merge_df.groupby('证券代码')['月收盘价'].pct_change()*100
#%%
# mv_df = pd.read_csv(r"C:\Users\zeng\Downloads\月个股回报率文件155136999\TRD_Mnth.csv")
# mv_df = mv_df.rename(columns={'Trdmnt':'trade_month','Msmvttl':'total_mv','Stkcd':'ts_code','Mclsprc':'close'})
# mv_df['ts_code'] = mv_df['ts_code'].apply(lambda x:'{:06d}'.format(x))
# def add_suffix(code):
#     if code.startswith('6'):
#         return code + '.SH'
#     elif code.startswith('8'):
#         return code + '.BJ'
#     elif code.startswith('4'):
#         return code + '.BJ'
#     else:
#         return code + '.SZ'

# # 使用apply方法应用上述函数
# mv_df['ts_code'] = mv_df['ts_code'].apply(add_suffix)
# # 收益率为百分比形式
# mv_df['stock_monthly_ret'] = mv_df.groupby('ts_code')['close'].pct_change()*100
# mv_df['next_month_ret'] =  mv_df.groupby('ts_code')['stock_monthly_ret'].shift(-1)
# mv_df['next_month_mv'] =  (mv_df.groupby('ts_code')['total_mv'].shift(-1))

# mv_df = mv_df.drop(columns=['close','total_mv'])

# month_data['trade_month'] = pd.to_datetime(month_data['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m')
# month_data['next_month_rf'] = month_data.groupby('ts_code')['rf'].shift(-1)
# month_data = pd.merge(month_data, mv_df,on=['trade_month','ts_code'],how='left')
# month_data['month'] = month_data['trade_month']
# month_data = month_data[month_data['trade_date']<20150101].drop(columns=['trade_month','year'])


# '''描述性统计'''
# print(month_data['NOA'].describe())

# positive_vars = ['BM', 'ADV', 'RD', 'GPOA']  # 正向指标
# negative_vars = ['beta', 'IVOL', 'ACC', 'NOA']  # 逆向指标
# big_vars = ['Safety','Cheapness','Quality']

# df = month_data.dropna(subset = positive_vars + negative_vars)

# def calculate_z_score(series, ascending):
#     # 对 series 进行排序，得到次序变量 r
#     r = series.rank(ascending=ascending)
#     # 计算次序变量的横截面均值和标准差
#     mu_r = r.mean()
#     sigma_r = r.std()
#     # 计算 z-score
#     z_score = (r - mu_r) / sigma_r
#     return z_score

# # 按月分组并计算每个指标的 z-score
# def process_monthly_data(group):
#     for var in positive_vars:
#         group[var + '_z'] = calculate_z_score(group[var], ascending=True)  # 正向指标从小到大排序，即升序
#     for var in negative_vars:
#         group[var + '_z'] = calculate_z_score(group[var], ascending=False)  # 逆向指标从大到小排序，即降序
#     return group

# def process_big_data(group):
#     for var in big_vars:
#         group[var + '_z'] = calculate_z_score(group[var], ascending=True)  # 正向指标从小到大排序
#     return group

# def process_b_score(group):
#     group['B_score_z'] = calculate_z_score(group['B_score'], ascending=True)  # 正向指标从小到大排序
#     return group

# # 按月分组并应用处理函数
# df = df.groupby('trade_date').apply(process_monthly_data).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

# df['Safety'] = (df['beta_z'] + df['IVOL_z']) 
# df['Cheapness'] = (df['BM_z'] + df['ADV_z'] + df['RD_z']) 
# df['Quality'] = (df['GPOA_z'] + df['ACC_z'] + df['NOA_z']) 

# df = df.groupby('trade_date').apply(process_big_data).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

# df['B_score'] = (df['Safety_z'] + df['Cheapness_z'] + df['Quality_z']) 

# df = df.groupby('trade_date').apply(process_b_score).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

# def single_var_test(var):
#     data = df[['ts_code','trade_date','next_month_ret',var,'next_month_mv']]
    
#     time_list = data['trade_date'].unique().tolist()
    
#     equal_weight_factor_return = pd.DataFrame()
#     for date in (time_list[1:]):
#         last_month = time_list[time_list.index(date)-1]
#         subset = data[data['trade_date'] == last_month].dropna()
#         subset['group'] = pd.qcut(subset[var], 5, labels=False, duplicates='drop')
#         group_avg = subset.groupby('group')['next_month_ret'].mean().tolist()
#         date_factor_ret = group_avg[4] - group_avg[0]
#         new_row = pd.DataFrame({'trade_date': [date], 'factor_return': [date_factor_ret],
#                                 'portfolio_05':[group_avg[4]],'portfolio_04':[group_avg[3]],
#                                 'portfolio_03':[group_avg[2]],'portfolio_02':[group_avg[1]],
#                                 'portfolio_01':[group_avg[0]]})
#         equal_weight_factor_return = pd.concat([equal_weight_factor_return, new_row], ignore_index=True)
#     equal_weight_factor_return['month'] = pd.to_datetime(equal_weight_factor_return['trade_date'],
#                                                          format='%Y%m%d').dt.strftime('%Y-%m')
   
#     value_weight_factor_return = pd.DataFrame()
#     for date in (time_list[1:]):
#         last_month = time_list[time_list.index(date)-1]
#         subset = data[data['trade_date'] == last_month].dropna()
#         subset['value_weight'] = subset['next_month_mv']/subset['next_month_mv'].sum()
#         subset['value_weight_ret'] = subset['value_weight']*subset['next_month_ret']
#         subset['group'] = pd.qcut(subset[var], 5, labels=False, duplicates='drop')
#         group_avg = subset.groupby('group')['value_weight_ret'].sum().tolist()
#         date_factor_ret = group_avg[4] - group_avg[0]
#         new_row = pd.DataFrame({'trade_date': [date], 'factor_return': [date_factor_ret],
#                                 'portfolio_05':[group_avg[4]],'portfolio_04':[group_avg[3]],
#                                 'portfolio_03':[group_avg[2]],'portfolio_02':[group_avg[1]],
#                                 'portfolio_01':[group_avg[0]]})
#         value_weight_factor_return = pd.concat([value_weight_factor_return, new_row], ignore_index=True)
#     value_weight_factor_return['month'] = pd.to_datetime(value_weight_factor_return['trade_date'],
#                                                          format='%Y%m%d').dt.strftime('%Y-%m')

#     return equal_weight_factor_return,value_weight_factor_return





# # equal_weight_factor_return = single_var_test('B_score_z')[1]

# # '''Fama三因子调整收益率'''
# # ff3_month_data = pd.read_csv(r"C:\Users\zeng\Desktop\巴菲特alpha\ff6因子月频数据.csv")[['month','MKT','SMB','HML','rf']]

# # equal_weight_factor_return = pd.merge(equal_weight_factor_return, ff3_month_data,on=['month'],how='left')

# # group_ret = []
# # for column in equal_weight_factor_return.columns[1:7]:
# #     y = equal_weight_factor_return[column]/100-equal_weight_factor_return['rf']
# #     X = sm.add_constant(equal_weight_factor_return[['MKT','SMB','HML']])
# #     model = sm.OLS(y, X).fit()
# #     alpha = model.params['const']*100
# #     group_ret.append({'group':column,'alpha':alpha,})
# #     if column == 'factor_return':
# #         alpha_t_value = model.tvalues['const']
# #         group_ret.append({'group':column,'long_short_t_value':alpha_t_value})
# #     else :
# #         continue
# # group_ret = pd.DataFrame(group_ret)   
# # print(group_ret)

# # from scipy.stats import ttest_1samp

# # # print(equal_weight_factor_return.iloc[:,1:-1].mean())
# # value_weight_factor_return = single_var_test('Quality_z')[1]
# # print(value_weight_factor_return.iloc[:,1:-1].mean())

# # # 假设检验时间序列数据的均值是否显著不同于0
# # t_stat, p_value = ttest_1samp(value_weight_factor_return['factor_return'], popmean=0)
# # print("t统计量:", t_stat)






















