import pandas as pd
import numpy as np
from scipy import stats

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
print(exsReturns.dtypes)
print(exsReturns.head())
print(exsReturns.tail())



#create new dataframe to hold summary statistics
resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
res = pd.DataFrame(columns = resNames)
# for i in range(1, len(indNames)):
#     res.iloc[i-1, 0] = np.power(exsReturns.iloc[:,i].prod(axis=0),1/nrow)
i = 1
addExs = exsReturns.copy()
addExs.iloc[:, 1:] = (exsReturns.iloc[:, 1:]/100 + 1)

print(addExs.head())

nyr = int(endDate[:4]) - int(begDate[:4])
print(nyr)
print(stats.gmean(addExs.iloc[:, 1:], axis=0)**(1.0/nyr) - 1)
