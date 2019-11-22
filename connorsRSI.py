import numpy as np
import pandas as pd

#RSI + updownLength + ROC

def RSI(period):
    pass

def updownRSI(updownLength, period):
    pass

def ROC(look_back_period):
    pass



if __name__=="__main__":

    data = pd.read_csv('aapl.csv', index_col='Date')
    print(data.head())