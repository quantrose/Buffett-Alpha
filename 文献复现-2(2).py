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
filtered_data['换手率'] = filtered_data['月个股交易股数']/filtered_data['流通股数']
time_list = month_data['trade_date'].unique().tolist() + [20041231]
filtered_data = filtered_data[filtered_data['月收盘日期'].isin(time_list)].reset_index().drop(columns=['index','日Amihud'])
merge_df = pd.merge(filtered_data, Aminhud_df,on=['证券代码','交易月份'],how='left')
# 4. 计算月收益率,收益率转为百分比形式
merge_df['月收益率'] = merge_df.groupby('证券代码')['月收盘价'].pct_change()*100
merge_df = merge_df.rename(columns={'月收盘日期':'trade_date','证券代码':'ts_code'})
month_data = pd.merge(month_data, merge_df[['trade_date','ts_code','流通市值对数','月Amihud','换手率','月收益率','月个股总市值']],on=['trade_date','ts_code'],how='left')

# 收益率为百分比形式
month_data['next_month_ret'] =  month_data.groupby('ts_code')['月收益率'].shift(-1)
month_data['next_month_mv'] =  (month_data.groupby('ts_code')['月个股总市值'].shift(-1))


month_data['next_month_rf'] = month_data.groupby('ts_code')['rf'].shift(-1)
month_data = month_data[month_data['trade_date']<20150101]

'''描述性统计'''
# print(month_data['beta'].describe())

positive_vars = ['BM', 'ADV', 'RD', 'GPOA']  # 正向指标
negative_vars = ['beta', 'IVOL', 'ACC', 'NOA']  # 逆向指标
big_vars = ['Safety','Cheapness','Quality']

df = month_data.dropna(subset = positive_vars + negative_vars)

def calculate_z_score(series, ascending):
    # 对 series 进行排序，得到次序变量 r
    r = series.rank(ascending=ascending)
    # 计算次序变量的横截面均值和标准差
    mu_r = r.mean()
    sigma_r = r.std()
    # 计算 z-score
    z_score = (r - mu_r) / sigma_r
    return z_score

# 按月分组并计算每个指标的 z-score
def process_monthly_data(group):
    for var in positive_vars:
        group[var + '_z'] = calculate_z_score(group[var], ascending=True)  # 正向指标从小到大排序，即升序
    for var in negative_vars:
        group[var + '_z'] = calculate_z_score(group[var], ascending=False)  # 逆向指标从大到小排序，即降序
    return group

def process_big_data(group):
    for var in big_vars:
        group[var + '_z'] = calculate_z_score(group[var], ascending=True)  # 正向指标从小到大排序
    return group

def process_b_score(group):
    group['B_score_z'] = calculate_z_score(group['B_score'], ascending=True)  # 正向指标从小到大排序
    return group

# 按月分组并应用处理函数
df = df.groupby('trade_date').apply(process_monthly_data).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

df['Safety'] = (df['beta_z'] + df['IVOL_z']) 
df['Cheapness'] = (df['BM_z'] + df['ADV_z'] + df['RD_z']) 
df['Quality'] = (df['GPOA_z'] + df['ACC_z'] + df['NOA_z']) 

df = df.groupby('trade_date').apply(process_big_data).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

df['B_score'] = (df['Safety_z'] + df['Cheapness_z'] + df['Quality_z']) 

df = df.groupby('trade_date').apply(process_b_score).drop(columns='trade_date').reset_index().drop(columns=['level_1'])

df['个股市值权重'] = df.groupby('trade_date')['月个股总市值'].transform(lambda x: x / x.sum())
df['next_month_VW_ret'] = df['个股市值权重']*df['月收益率']

'''单变量分组检验'''
def single_var_test(var):
    data = df[['ts_code','trade_date','next_month_ret',var,'next_month_mv']]
    
    time_list = data['trade_date'].unique().tolist()
    
    equal_weight_factor_return = pd.DataFrame()
    for date in (time_list[1:]):
        last_month = time_list[time_list.index(date)-1]
        subset = data[data['trade_date'] == last_month].dropna()
        subset['group'] = pd.qcut(subset[var], 5, labels=False, duplicates='drop')
        group_avg = subset.groupby('group')['next_month_ret'].mean().tolist()
        date_factor_ret = group_avg[4] - group_avg[0]
        new_row = pd.DataFrame({'trade_date': [date], 'factor_return': [date_factor_ret],
                                'portfolio_05':[group_avg[4]],'portfolio_04':[group_avg[3]],
                                'portfolio_03':[group_avg[2]],'portfolio_02':[group_avg[1]],
                                'portfolio_01':[group_avg[0]]})
        equal_weight_factor_return = pd.concat([equal_weight_factor_return, new_row], ignore_index=True)
    equal_weight_factor_return['month'] = pd.to_datetime(equal_weight_factor_return['trade_date'],
                                                          format='%Y%m%d').dt.strftime('%Y-%m')
   
    value_weight_factor_return = pd.DataFrame()
    for date in (time_list[1:]):
        last_month = time_list[time_list.index(date)-1]
        subset = data[data['trade_date'] == last_month].dropna()
        subset['value_weight'] = subset['next_month_mv']/subset['next_month_mv'].sum()
        subset['value_weight_ret'] = subset['value_weight']*subset['next_month_ret']
        subset['group'] = pd.qcut(subset[var], 5, labels=False, duplicates='drop')
        group_avg = subset.groupby('group')['value_weight_ret'].sum().tolist()
        date_factor_ret = group_avg[4] - group_avg[0]
        new_row = pd.DataFrame({'trade_date': [date], 'factor_return': [date_factor_ret],
                                'portfolio_05':[group_avg[4]],'portfolio_04':[group_avg[3]],
                                'portfolio_03':[group_avg[2]],'portfolio_02':[group_avg[1]],
                                'portfolio_01':[group_avg[0]]})
        value_weight_factor_return = pd.concat([value_weight_factor_return, new_row], ignore_index=True)
    value_weight_factor_return['month'] = pd.to_datetime(value_weight_factor_return['trade_date'],
                                                          format='%Y%m%d').dt.strftime('%Y-%m')

    return equal_weight_factor_return,value_weight_factor_return


'''双变量分组检验（等权收益率）'''
def EW_double_var_test(var):
    df['市值分组'] = df.groupby('trade_date')['var'].transform(lambda x: pd.qcut(x, 3, labels=['small','middle','big'], duplicates='drop'))
    df['市值后再分组'] = df.groupby(['trade_date','市值分组'])['B_score_z'].transform(lambda x: pd.qcut(x, 5, labels=['01','02','03','04','05'], duplicates='drop'))
    df['总分组'] = df['市值分组'].astype(str) +'_'+ df['市值后再分组'].astype(str)
    month_ret = df.groupby(['trade_date','总分组'])['next_month_ret'].mean().dropna().reset_index()
    
    
    
    month_ret = month_ret.sort_values(['总分组','trade_date'])
    month_ret['组合收益率'] = month_ret.groupby('总分组')['next_month_ret'].shift(1)
    month_ret['month'] = pd.to_datetime(month_ret['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m')
    # equal_weight_factor_return = single_var_test('B_score_z')[1]
    
    '''Fama三因子调整收益率'''
    ff3_month_data = pd.read_csv(r"C:\Users\zeng\Desktop\数据\ff6因子月频数据.csv")[['month','MKT','SMB','HML','rf']]
    
    equal_weight_factor_return = pd.merge(month_ret, ff3_month_data,on=['month'],how='left').dropna()
    
    # 计算多空收益率
    long_short_portfolio = []
    for date in equal_weight_factor_return['trade_date'].unique():
        subset = equal_weight_factor_return[equal_weight_factor_return['trade_date']==date]
        big_longshort_ret = (subset[subset['总分组']=='big_05']['组合收益率'].values - subset[subset['总分组']=='big_01']['组合收益率'].values)[0]
        middle_longshort_ret = (subset[subset['总分组']=='middle_05']['组合收益率'].values - subset[subset['总分组']=='middle_01']['组合收益率'].values)[0]
        small_longshort_ret = (subset[subset['总分组']=='small_05']['组合收益率'].values - subset[subset['总分组']=='small_01']['组合收益率'].values)[0]
        long_short_portfolio.append({'trade_date':date,'big':big_longshort_ret,'middle':middle_longshort_ret,'small':small_longshort_ret})
    long_short_portfolio = pd.DataFrame(long_short_portfolio)
    long_short_portfolio['month'] = pd.to_datetime(long_short_portfolio['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m')
    
    long_short_portfolio = pd.merge(long_short_portfolio, ff3_month_data,on=['month'],how='left')
    LS_t_test = []
    for column in long_short_portfolio.columns[1:4]:
        y = long_short_portfolio[column]/100-long_short_portfolio['rf']
        X = sm.add_constant(long_short_portfolio[['MKT','SMB','HML']])
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']*100
        alpha_t_value = model.tvalues['const']
        LS_t_test.append({'group':column,'alpha':alpha,'long_short_t_value':alpha_t_value})
    
    LS_t_test = pd.DataFrame(LS_t_test)   
    print(LS_t_test)
    
    def cal_alpha(group):
        y = group['组合收益率']/100-group['rf']
        X = sm.add_constant(group[['MKT','SMB','HML']])
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']*100
        df = pd.DataFrame(index=group['总分组'].unique(),data={'alpha':alpha})
        return df
    
    alpha_df = equal_weight_factor_return.groupby('总分组').apply(cal_alpha)
    print(alpha_df)
    
    
'''双变量分组检验（市值加权收益率）'''
def EW_double_var_test(var):
    df['市值分组'] = df.groupby('trade_date')[var].transform(lambda x: pd.qcut(x, 3, labels=['small','middle','big'], duplicates='drop'))
    df['市值后再分组'] = df.groupby(['trade_date','市值分组'])['B_score_z'].transform(lambda x: pd.qcut(x, 5, labels=['01','02','03','04','05'], duplicates='drop'))
    df['总分组'] = df['市值分组'].astype(str) +'_'+ df['市值后再分组'].astype(str)
    month_ret = df.groupby(['trade_date','总分组'])['next_month_VW_ret'].sum().dropna().reset_index()
    
    
    
    month_ret = month_ret.sort_values(['总分组','trade_date'])
    month_ret['组合收益率'] = month_ret.groupby('总分组')['next_month_VW_ret'].shift(1)
    month_ret['month'] = pd.to_datetime(month_ret['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m')
    # equal_weight_factor_return = single_var_test('B_score_z')[1]
    
    '''Fama三因子调整收益率'''
    ff3_month_data = pd.read_csv(r"C:\Users\zeng\Desktop\数据\ff6因子月频数据.csv")[['month','MKT','SMB','HML','rf']]
    
    equal_weight_factor_return = pd.merge(month_ret, ff3_month_data,on=['month'],how='left').dropna()
    
    # 计算多空收益率
    long_short_portfolio = []
    for date in equal_weight_factor_return['trade_date'].unique():
        subset = equal_weight_factor_return[equal_weight_factor_return['trade_date']==date]
        big_longshort_ret = (subset[subset['总分组']=='big_05']['组合收益率'].values - subset[subset['总分组']=='big_01']['组合收益率'].values)[0]
        middle_longshort_ret = (subset[subset['总分组']=='middle_05']['组合收益率'].values - subset[subset['总分组']=='middle_01']['组合收益率'].values)[0]
        small_longshort_ret = (subset[subset['总分组']=='small_05']['组合收益率'].values - subset[subset['总分组']=='small_01']['组合收益率'].values)[0]
        long_short_portfolio.append({'trade_date':date,'big':big_longshort_ret,'middle':middle_longshort_ret,'small':small_longshort_ret})
    long_short_portfolio = pd.DataFrame(long_short_portfolio)
    long_short_portfolio['month'] = pd.to_datetime(long_short_portfolio['trade_date'],format='%Y%m%d').dt.strftime('%Y-%m')
    
    long_short_portfolio = pd.merge(long_short_portfolio, ff3_month_data,on=['month'],how='left')
    LS_t_test = []
    for column in long_short_portfolio.columns[1:4]:
        y = long_short_portfolio[column]/100-long_short_portfolio['rf']
        X = sm.add_constant(long_short_portfolio[['MKT','SMB','HML']])
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']*100
        alpha_t_value = model.tvalues['const']
        LS_t_test.append({'group':column,'alpha':alpha,'long_short_t_value':alpha_t_value})
    
    LS_t_test = pd.DataFrame(LS_t_test)   
    print(LS_t_test)
    
    def cal_alpha(group):
        y = group['组合收益率']/100-group['rf']
        X = sm.add_constant(group[['MKT','SMB','HML']])
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']*100
        df = pd.DataFrame(index=group['总分组'].unique(),data={'alpha':alpha})
        return df
    
    alpha_df = equal_weight_factor_return.groupby('总分组').apply(cal_alpha)
    print(alpha_df)
    
#%%
'''Fama回归'''
model_0_y = (df['月收益率']/100).fillna(0)
model_0_X = sm.add_constant(df['B_score_z'])
model_1_X = sm.add_constant(df['Safety_z'])
model_2_X = sm.add_constant(df['Cheapness_z'])
model_3_X = sm.add_constant(df['Quality_z'])
model_coef = sm.OLS(model_0_y,model_0_X).fit().params[1]
model_t_value = sm.OLS(model_0_y,model_0_X).fit().tvalues[1]
print(model_coef)
print(model_t_value)




















