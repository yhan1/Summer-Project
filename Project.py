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
	exsReturns = excessReturns(ind, rf)
	
	classList = createClass(exsReturns)
	sumTable = summaryStat(ind, rf)
	#print(sumTable)
	writer = pd.ExcelWriter("output.xlsx")
	sumTable.to_excel(writer, "Sheet1")
	writer.save()


	indNames = list(exsReturns)


	#for indIndex in range(len(exsReturns)):
# 		#indIndex: 0 to 29 inclusive
# 		(ynorm, Xnorm) = normalize(classList, indIndex)
# 		lasso = lassoM(ynorm, Xnorm)
# #lassoM(exsReturns, indIndex, sumTable) #lasso for variable selection
# 		xIndex = np.nonzero(lasso.coef_)[0]
# 		Xlin = Xnorm[Xnorm.columns[xIndex]]
# 		print(Xlin)
# 		ols = linearM(ynorm, Xlin)
# 		print(indIndex)
# 		print(xIndex)
# 		print(ols.coef_)
	indIndex = 0
	(ynorm, Xnorm) = normalize(classList, indIndex)
	lasso = lassoM(ynorm, Xnorm)
#lassoM(exsReturns, indIndex, sumTable) #lasso for variable selection
	xIndex = np.nonzero(lasso.coef_)[0]
	Xlin = Xnorm[Xnorm.columns[xIndex]]
	print(Xlin)
	ols = linearM(ynorm, Xlin)
	print(indIndex)
	print(xIndex)
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
	rf["rf"] = rf["rf"] * 100 #all returns in percentages

	return (ind, rf)

def createClass(excessReturns):
	classList = []
	colNames = list(excessReturns)
	for i in range(len(colNames)):
		a = IndExs(i, excessReturns)
		classList.append(a)
	return classList


def excessReturns(ind, rf):
	exsReturns = ind.copy()
	indNames = list(ind)
	for i in range(1, len(indNames)):
		exsReturns.iloc[:, i] = ind.iloc[:, i] - rf.iloc[:, 1]
	return exsReturns.iloc[:, 1:] #leave out dates


def summaryStat(ind, rf):
	exsReturns = excessReturns(ind, rf)

	#create new dataframe to hold summary statistics
	resNames = ["Ann mean", "Ann vol", "Minimum", "Maximum", "Ann Sharpe"]
	res = pd.DataFrame(columns = resNames)

	nrow = rf.count()[0]

	#ann mean, vol, min, max, sharpe
	res.iloc[:, 0] = (np.prod(exsReturns.iloc[:, 1:]/100 + 1, axis=0) ** (1/(nrow/12)) -1)*100
	#res.iloc[:, 0] = np.mean(exsReturns.iloc[:, 1:], axis=0)
	res.iloc[:, 1] = (np.std(exsReturns.iloc[:, 1:], axis=0) * (12 ** 0.5))
	res.iloc[:, 2] = np.amin(exsReturns.iloc[:, 1:], axis=0)
	res.iloc[:, 3] = np.amax(exsReturns.iloc[:, 1:], axis=0)
	res.iloc[:, 4] = res.iloc[:, 0] / res.iloc[:, 1]

	return res


class IndExs(object):
	def __init__(self, indIndex, df): #df = excess returns dataframe
		self.exs = df.iloc[:, indIndex]
		self.n = len(self.exs)
		self.y = self.exs[1:]      #does not include first month (1959-12), for lagged return
		self.df = df
		self.X = df.iloc[:-1, ]    #leave out last month (2016-12) for lagged return
		self.mean = (np.prod(self.exs/100 + 1, axis=0) ** (1/(len(self.exs)/12)) -1)*100
		self.sd = (np.std(self.exs, axis=0) * (12 ** 0.5))
		
		self.ynorm = (self.y - self.mean) / self.sd
		self.Xnorm = pd.DataFrame(columns = list(df))

		self.min = np.amin(self.exs, axis=0)
		self.max = np.amax(self.exs, axis=0)
		self.sharpe = self.mean / self.sd

def normalize(classList, indIndex):
	y = classList[indIndex]
	ynorm = (y.exs - y.mean) / y.sd
	Xnorm = pd.DataFrame(columns = list(y.df))
	i = 0
	for ind in classList:
		#xnorm = (ind.exs - ind.mean) / ind.sd
		Xnorm.iloc[:, i] = (ind.exs - ind.mean) / ind.sd
		i += 1
	return (ynorm, Xnorm)

def lassoM(ynorm, Xnorm):
	lasso = LassoLarsIC(criterion = "aic")
	#print("Industry = ", colNames[indIndex])
	#normalize/standardize the returns

	lasso.fit(Xnorm, ynorm)
	return lasso


def linearM(ylin, Xlin):
	lin = LinearRegression()
	lin.fit(Xlin, ylin)
	return lin


# lasso = LassoLarsIC(criterion = "aic")
# lasso.fit(exsReturns.iloc[:, 1:])



main()