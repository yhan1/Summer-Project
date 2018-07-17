import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC, LinearRegression, LassoCV, LassoLarsCV
from sklearn import preprocessing
import networkx as nx

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():

	indPath = "Data/30_Industry_Portfolios.CSV"
	rfPath = "Data/rf.csv"
	begDate = "195912"
	endDate = "201612"
	(ind, rf) = loadData(indPath, rfPath, begDate, endDate)
	exsReturns = excessReturns(ind, rf) 
	nrow = rf.count()[0]
	# writer = pd.ExcelWriter("excess.xlsx")
	# exsReturns.to_excel(writer, "Sheet1")
	# writer.save()


	#creat list of industry classes
	classList = createClass(exsReturns.iloc[:, 1:]) # no date
	sumTable = summaryStat(exsReturns.iloc[:, 1:]) # no date
	# print(sumTable)
	# writer = pd.ExcelWriter("output.xlsx")
	# sumTable.to_excel(writer, "Sheet1")
	# writer.save()


	indNames = list(exsReturns.iloc[:, 1:])
	m = ["aic", "bic", "LassoCV", "LassoLarsCV"] #aic and bic is using LassoLarsIC

	# for method in m:
	# 	lassoResults = regression(exsReturns, method)
		# writer = pd.ExcelWriter(method + ".xlsx")
		# lassoResults.to_excel(writer, "Sheet1")
		# writer.save()

	df = exsReturns #with date
	fullPeriodResult = regression(df) # use aic
	
	# expanding period L/S portfolio construction
	startRow = 0
	endRow = df.loc[df["Date"] == 196912].index[0]
	lastRow = df.loc[df["Date"] == 201612].index[0]
	#for e in range(endRow, lastRow):
	e = endRow
	print("startRow = ", startRow, "endRow = ", e)
	B = regression(df, endRow = e)
	X = df.iloc[e + 1, 1:] # no date, next time period
	yhat = np.dot(B, X)
	yhat = pd.DataFrame(yhat, index = B.index, columns = ["yhat"])
	print(yhat)
	yhat.sort_values(by = ["yhat"], ascending = True, inplace = True)
	#print(yhat)
	print(df.iloc[e + 1, 1:])






	


def regression(df, method="aic", startRow = 0, endRow = 684): #inclusive rows 

	indNames = list(df.iloc[:, 1:])

	lassoResults = pd.DataFrame(np.zeros((len(indNames), len(indNames))), columns = indNames, index = indNames)
	df = df.iloc[startRow:endRow + 1, 1:] # no date

	for indIndex in range(len(indNames)):
		X = df.iloc[:-1, :] 
		y = df.iloc[1:, indIndex]

		lasso = lassoM(X, y, method)
		xIndex = np.nonzero(lasso.coef_)[0]
		Xlin = X[X.columns[xIndex]]
		if xIndex != []:
			ols = linearM(Xlin, y)
			j = 0
			for i in xIndex:
				lassoResults.iloc[indIndex, i] = ols.coef_[j]
				j += 1
	return lassoResults

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
	rf["rf"] = rf["rf"] * 100 #all returns in percentages

	return (ind, rf)

def excessReturns(ind, rf):
	exsReturns = ind.copy()
	indNames = list(ind)
	for i in range(1, len(indNames)):
		exsReturns.iloc[:, i] = ind.iloc[:, i] - rf.iloc[:, 1]
	return exsReturns

def createClass(excessReturns):
	classList = []
	colNames = list(excessReturns.iloc[:, 1:])
	for i in range(len(colNames)):
		a = IndExs(i, excessReturns)
		classList.append(a)
	return classList


def summaryStat(exsReturns):


	#create new dataframe to hold summary statistics
	resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
	res = pd.DataFrame(columns = resNames)


	#ann mean, vol, min, max, sharpe
	#res.iloc[:, 0] = (np.prod(exsReturns/100 + 1, axis=0) ** (1/(nrow/12)) -1)*100
	res.iloc[:, 0] = np.mean(exsReturns, axis=0) * 12
	res.iloc[:, 1] = (np.std(exsReturns, axis=0) * (12 ** 0.5))
	res.iloc[:, 2] = np.amin(exsReturns, axis=0)
	res.iloc[:, 3] = np.amax(exsReturns, axis=0)
	res.iloc[:, 4] = res.iloc[:, 0] / res.iloc[:, 1]

	return res


class IndExs(object):
	def __init__(self, indIndex, df): #df = excess returns dataframe
		#self.exs = df.iloc[:, indIndex]
		#self.n = len(self.exs)
		#self.y = self.exs[1:]      #does not include first month (1959-12), for lagged return
		#self.X = df.iloc[:-1, ]    #leave out last month (2016-12) for lagged return
		self.df = df
		self.name = list(df)[indIndex]

def normalize(classList, indIndex):
	yInd = classList[indIndex]
	ynorm = norm(yInd.y)
	Xnorm = pd.DataFrame(columns = list(yInd.df))
	i = 0
	for ind in classList:
		r = ind.exs.iloc[:-1] # leave out last month for x values
		Xnorm.iloc[:, i] = norm(r)
		i += 1
	return (Xnorm, ynorm)


def norm(series):
	return (series - np.mean(series) / np.std(series))

def lassoM(X, y, method):
	if (method == "aic") or (method == "bic"):
		lasso = LassoLarsIC(criterion = method, normalize = True)
		lasso.fit(X, y)
	elif method == "LassoCV": 
		lasso = LassoCV(cv = 20).fit(X, y)
	
	elif method == "LassoLarsCV": 
		lasso = LassoLarsCV(cv = 20).fit(X, y)
	
	#print(lasso.alpha_)
	return lasso

def linearM(X, y):
	lin = LinearRegression()
	lin.fit(X, y)
	return lin


main()