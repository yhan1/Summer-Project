import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC

#read in data
ind = pd.read_csv("Data/30_Industry_Portfolios.CSV", skiprows = 11)
rf = pd.read_csv("Data/rf.csv")
rf.dateff = rf.dateff // 100

#rename column
ind.rename(columns={'Unnamed: 0':'Date'}, inplace=True )

#list with all column names
indNames = list(ind)

#only select certain dates 
begDate = "195912"
endDate = "201612"
rf = rf.loc[(rf["dateff"] >= int(begDate)) & (rf["dateff"] <= int(endDate))]
nrow = rf.count()[0]
startRow = (ind[ind.Date == begDate].index)[0]
endRow = (ind[ind.Date == endDate].index)[0]
ind = ind.iloc[startRow:endRow + 1]
ind = ind.reset_index(drop=True)
rf = rf.reset_index(drop=True)

#convert returns to float and year to int
ind[indNames[1:]] = ind[indNames[1:]].astype(float)
ind[indNames[0]] = ind[indNames[0]].astype(int)

#compute excess returns
exsReturns = ind.copy()
for i in range(1, len(indNames)):
    exsReturns.iloc[:, i] = ind.iloc[:, i] - rf.iloc[:, 1]


# print(ind.head())
# print(ind.tail())
# print(rf.head())
# print(rf.tail())
# print(ind.shape)
# print(rf.shape)
#print(exsReturns.dtypes)
print(exsReturns.head())
print(exsReturns.tail())

#create new dataframe to hold summary statistics
resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
res = pd.DataFrame(columns = resNames)

nyr = int(endDate[:4]) - int(begDate[:4])
print(nyr)
#print((stats.gmean(addExs.iloc[:, 1:], axis=0) - 1)*100)

#ann mean, vol, min, max, sharpe
res.iloc[:, 0] = (np.prod(exsReturns.iloc[:, 1:]/100 + 1, axis=0) ** (1/(nrow/12)) -1)*100
res.iloc[:, 1] = (np.std(exsReturns.iloc[:, 1:], axis=0) * (12 ** 0.5))
res.iloc[:, 2] = np.amin(exsReturns.iloc[:, 1:], axis=0)
res.iloc[:, 3] = np.amax(exsReturns.iloc[:, 1:], axis=0)
res.iloc[:, 4] = res.iloc[:, 0] / res.iloc[:, 1]

print(res.head())
print(res.tail())

lasso = LassoLarsIC(criterion = "aic")
lasso.fit(exsReturns.iloc[:, 1:])

