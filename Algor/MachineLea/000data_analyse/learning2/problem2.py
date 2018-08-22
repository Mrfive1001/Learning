import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing

election = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning2\2012-general-election-romney-vs-obama.csv')
election['Start Date'], election['End Date'] = pd.to_datetime(election['Start Date']), pd.to_datetime(
    election['End Date'])
temp = election[election['Start Date'] > datetime.datetime(2011, 10, 31)].copy()
temp = temp[temp['Start Date'] < datetime.datetime(2011, 11, 30)]
print(len(temp.index))
