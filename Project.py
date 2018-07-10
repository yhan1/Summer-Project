import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn import preprocessing


def main():
	indPath = "Data/30_Industry_Portfolios.CSV"
	rfPath = "Data/rf.csv"
	begDate = "195912"
	endDate = "201612"
	(ind, rf) = loadData(indPath, rfPath, begDate, endDate)
	exsReturns = excessReturns(ind, rf) #no date
	writer = pd.ExcelWriter("excess.xlsx")
	exsReturns.to_excel(writer, "Sheet1")
	writer.save()


	#creat list of industry classes
	classList = createClass(exsReturns)
	sumTable = summaryStat(ind, rf)
	print(sumTable)
	# writer = pd.ExcelWriter("output.xlsx")
	# sumTable.to_excel(writer, "Sheet1")
	# writer.save()


	indNames = list(exsReturns)
	
	for indIndex in range(len(indNames)):
		#(Xnorm, ynorm) = normalize(classList, indIndex)
		X = classList[indIndex].X
		y = classList[indIndex].y
		lasso = lassoM(X, y)

		xIndex = np.nonzero(lasso.coef_)[0]
		Xlin = X[X.columns[xIndex]]
		print("Industry = ", indNames[indIndex])
		for i in xIndex:
			print (indNames[i])
		# print(lasso.coef_)
		#print(ols.coef_)


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
	return exsReturns.iloc[:, 1:] #leave out dates

def createClass(excessReturns):
	classList = []
	colNames = list(excessReturns)
	for i in range(len(colNames)):
		a = IndExs(i, excessReturns)
		classList.append(a)
	return classList


def summaryStat(ind, rf):
	exsReturns = excessReturns(ind, rf)

	#create new dataframe to hold summary statistics
	resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
	res = pd.DataFrame(columns = resNames)

	nrow = rf.count()[0]

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
		self.exs = df.iloc[:, indIndex]
		self.n = len(self.exs)
		self.y = self.exs[1:]      #does not include first month (1959-12), for lagged return
		self.X = df.iloc[:-1, ]    #leave out last month (2016-12) for lagged return
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

def lassoM(X, y):
	lasso = LassoLarsIC(criterion = "aic", normalize = True)
	lasso.fit(X, y)
	print(lasso.alpha_)
	return lasso

def linearM(X, y):
	lin = LinearRegression()
	lin.fit(X, y)
	return lin



main()