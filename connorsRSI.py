import numpy as np
import pandas as pd

#RSI + updownLength + ROC

def RSI(data, period):
    '''
    100 - 100/(1 + avggain/avgloss)
    '''
    pro_los = data.diff().dropna()
    profit_days = pro_los.copy()
    loss_days = np.abs(pro_los.copy())
    profit_days[pro_los < 0] = 0
    loss_days[pro_los > 0] = 0
    
    avg_rolling_gain = profit_days.rolling(period).mean()
    avg_rolling_loss = loss_days.rolling(period).mean()
    rsi = 100 - 100/(1 + avg_rolling_gain/avg_rolling_loss)
    
    # print(rsi)
    return rsi

def updownLength(data):
    '''
    return a series of up-down streak values
    '''
    streaks = pd.Series(data=np.zeros(data.size))
    curr_streak = 0
    
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            flag = 'up'
            if flag != 'up':
                curr_streak = 0
            streaks[i] = curr_streak + 1
        elif data[i] < data[i-1]:
            flag = 'down'
            if flag != 'down':
                curr_streak = 0
            streaks[i] = curr_streak - 1
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

    #daily appl data for last 5 years
    data = pd.read_csv('aapl.csv', index_col='Date')
    prices = [pr for pr in data.Close]
    prices = pd.Series(prices)

    #PARAMS
    RSI_PERIOD = 3
    RSI_UPDOWN_PERIOD = 2
    ROC_LB_PERIOD = 100

    # defining connorsRSI
    crsi = (RSI(prices, RSI_PERIOD) + RSI(updownLength(prices), RSI_UPDOWN_PERIOD) + ROC(ROC_LB_PERIOD))/3

