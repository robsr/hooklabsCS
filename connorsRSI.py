import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
sns.set_style('darkgrid')

'''
all the definitions taken from here
https://www.tradingview.com/wiki/Connors_RSI_(CRSI)#CALCULATION
'''

def calcNA(series):
    return series.isna().sum()

def RSI(data, period):
    '''
    100 - 100/(1 + avggain/avgloss)
    '''
    # print(data[:10])
    pro_los = data.diff().dropna()
    profit_days = pro_los.copy()
    loss_days = np.abs(pro_los.copy())
    profit_days[pro_los < 0] = 0
    loss_days[pro_los > 0] = 0
    # print(loss_days)

    avg_rolling_gain = profit_days.rolling(period).mean()
    avg_rolling_loss = loss_days.rolling(period).mean()
    rsi = 100 - 100/(1 + avg_rolling_gain/avg_rolling_loss)
    
    return rsi

def updownLength(data):
    '''
    return a series of up-down streak values
    '''
    streaks = pd.Series(data=np.zeros(data.size))
    curr_streak = 0
    flag='neutral'

    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            if flag != 'up':
                curr_streak = 0
            curr_streak += 1
            streaks[i] = curr_streak
            flag = 'up'
        elif data[i] < data[i-1]:
            if flag != 'down':
                curr_streak = 0
            curr_streak -= 1
            streaks[i] = curr_streak
            flag = 'down'
        else:
            curr_streak = 0
            streaks[i] = curr_streak
            flag = 'neutral'
    return streaks



def ROC(data, lb_period):
    '''
    %of no. of values within the last look back period that are below the current price change percentage
    '''
    diff_data = data.diff(lb_period)    
    old_data = data.shift(periods=lb_period, axis=0)
    roc = pd.Series(diff_data/old_data)
    return roc

if __name__=="__main__":

    #daily appl data for last 1 year
    ## FROM ['2018-11-21'] to ['2019-11-21']
    data = pd.read_csv('AAPL.csv', index_col='Date')

    prices = [pr for pr in data.Close]
    prices = pd.Series(prices)

    #PARAMS
    RSI_PERIOD = 3
    RSI_UPDOWN_PERIOD = 2
    ROC_LB_PERIOD = 50



    # #3components
    C1 = RSI(prices, RSI_PERIOD)
    C2 = RSI(updownLength(prices), RSI_UPDOWN_PERIOD)    
    C3 = ROC(prices, ROC_LB_PERIOD)

    # defining connorsRSI
    crsi = (C1 + C2 + C3)/3

    #top plot the results it only makes sense after the first ROC_LB_period whichi is max among all the time periods

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data.index.values, prices, label='price')
    ax.plot(data.index.values, crsi, label='CRSI')
    ax.set(xlabel='Date')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    
    plt.legend()
    plt.show()