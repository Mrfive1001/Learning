import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing

exprs = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning2\exprs_GSE5859.csv',
                    sep=',', index_col=0)
sampleinfo = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning2\sampleinfo_GSE5859.csv',
                         sep=',')

a = list(sampleinfo.filename)
b = list(exprs.columns)
matchIndex = [b.index(x) for x in a]

exprs = exprs.iloc[:, matchIndex]
print(exprs.head())
print(sampleinfo.head())

sampleinfo.date = pd.to_datetime(sampleinfo.date)
sampleinfo['year'] = [i.year for i in sampleinfo.date]
sampleinfo['month'] = [i.month for i in sampleinfo.date]

special = datetime.datetime(2002, 10, 31)
sampleinfo['elapsedInDays'] = [i - special for i in sampleinfo.date]
print(sampleinfo.head())

CEUsample = sampleinfo[sampleinfo.ethnicity == 'CEU'].copy()
print(CEUsample.head())
CEUexprs = exprs.iloc[:, np.arange(len(exprs.columns))[sampleinfo.ethnicity == 'CEU']].copy()
pd.DataFrame(preprocessing.minmax_scale(CEUexprs), index=CEUexprs.index, columns=CEUexprs.columns)
