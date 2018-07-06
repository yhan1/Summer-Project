import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC, LinearRegression


def main():
	indPath = "Data/30_Industry_Portfolios.CSV"
	rfPath = "Data/rf.csv"
	begDate = "195912"
	endDate = "201612"
	(ind, rf) = loadData(indPath, rfPath, begDate, endDate)
	table1 = summaryStat(ind, rf)
	#print(table1)

	#model
	exsReturns = excessReturns(ind, rf)
	indIndex = 1 # 1 to 30 inclusive
	lasso = lassoM(exsReturns, indIndex) #lasso for variable selection
	xIndex = np.nonzero(lasso.coef_)[0]
	ols = linearM(exsReturns, indIndex, xIndex)
	print(ols.coef_)


def loadData(indPath, rfPath, begDate, endDate):
	#read in data
	ind = pd.read_csv(indPath, skiprows = 11)
	rf = pd.read_csv(rfPath)
	rf.dateff = rf.dateff // 100

	#rename column
	ind.rename(columns={'Unnamed: 0':'Date'}, inplace=True )

	#only select certain dates 
	
	rf = rf.loc[(rf["dateff"] >= int(begDate)) & (rf["dateff"] <= int(endDate))]
	startRow = (ind[ind.Date == begDate].index)[0]
	endRow = (ind[ind.Date == endDate].index)[0]
	ind = ind.iloc[startRow:endRow + 1]
	ind = ind.reset_index(drop=True)
	rf = rf.reset_index(drop=True)

	indNames = list(ind)

	#convert returns to float and year to int
	ind[indNames[1:]] = ind[indNames[1:]].astype(float)
	ind[indNames[0]] = ind[indNames[0]].astype(int)

	return (ind, rf)


def excessReturns(ind, rf):
	exsReturns = ind.copy()
	indNames = list(ind)
	for i in range(1, len(indNames)):
		exsReturns.iloc[:, i] = ind.iloc[:, i] - rf.iloc[:, 1]
	return exsReturns


def summaryStat(ind, rf):
	exsReturns = excessReturns(ind, rf)

	#create new dataframe to hold summary statistics
	resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
	res = pd.DataFrame(columns = resNames)

	nrow = rf.count()[0]

	#ann mean, vol, min, max, sharpe
	res.iloc[:, 0] = (np.prod(exsReturns.iloc[:, 1:]/100 + 1, axis=0) ** (1/(nrow/12)) -1)*100
	res.iloc[:, 1] = (np.std(exsReturns.iloc[:, 1:], axis=0) * (12 ** 0.5))
	res.iloc[:, 2] = np.amin(exsReturns.iloc[:, 1:], axis=0)
	res.iloc[:, 3] = np.amax(exsReturns.iloc[:, 1:], axis=0)
	res.iloc[:, 4] = res.iloc[:, 0] / res.iloc[:, 1]

	return res


class IndExs(object):
	def __init__(self, indIndex, df):
		self.series = df.iloc[:, indIndex]
		self.n = len(self.series)
		# self.y = df.iloc[1:, indIndex] #does not include first month (1959-12), for lagged return
		# self.X = df.iloc[:-1, ] #leave out last month (2016-12) for lagged return

def lassoM(exsReturns, indIndex):
	lasso = LassoLarsIC(criterion = "aic")
	print("Industry = ", list(exsReturns)[indIndex])
	y = exsReturns.iloc[1:, indIndex]
	X = exsReturns.iloc[:-1]
	lasso.fit(X, y)
	return lasso


def linearM(exsReturns, indIndex, xIndex):
	lin = LinearRegression()
	tmp = exsReturns.iloc[:-1]
	X = tmp[tmp.columns[xIndex]]
	y = exsReturns.iloc[1:, indIndex]
	lin.fit(X, y)
	return lin





# lasso = LassoLarsIC(criterion = "aic")
# lasso.fit(exsReturns.iloc[:, 1:])



main()