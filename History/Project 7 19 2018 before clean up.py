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
	(inter, fullPeriodResult)= OLSlassoRegression(df) # use aic
	# writer = pd.ExcelWriter("aic" + ".xlsx")
	# fullPeriodResult.to_excel(writer, "Sheet1")
	# writer.save()
	#print(fullPeriodResult)
	
	# expanding period L/S portfolio construction
	startRow = 0
	endRow = df.loc[df["Date"] == 196912].index[0]
	lastRow = df.loc[df["Date"] == 201612].index[0]
	periodR = pd.DataFrame(np.zeros(lastRow - endRow))
	for e in range(endRow, lastRow):
		#print(e)
		out = linRegression(df, endRow = e, mode="predict")
		out.sort_values(by = ["yPred"], ascending = True, inplace = True)
		#print("Mean positive", np.mean(out[out > 0.0]))

		#print(out)

		#after sorted returns, long top quintile, and short bottom quintile
		bottomInd = out.iloc[:5, :].index #gives industries
		topInd = out.iloc[-5:, :].index
		
		bottomR = df.loc[endRow + 1, bottomInd] #realized return
		topR = df.loc[endRow + 1, topInd]
		if np.average(topR) > 0: print("POSITIVE RETURNNNNNNNNN")
		#print("predicted returns \n", out)
		print(topR, "\n", bottomR, "\n")
		print(df.iloc[e + 1, 0], np.round(np.average(topR)), np.round(np.average(bottomR)), np.round(np.average(topR) - np.average(bottomR)))
		periodR.iloc[e - endRow, :] = np.mean(topR) - np.mean(bottomR)
		# print(bottomR)
		# print(topR)
		# print(df.loc[endRow + 1, :].sort_values())
		# print(df.loc[endRow + 1, "Date"])
		#print(df.iloc[e + 1, 1:])
	print(np.mean(periodR) * 12)



def OLSlassoRegression(df, method="aic", startRow = 0, endRow = 684, mode="fit"): #inclusive rows 
	indNames = list(df.iloc[:, 1:])
	#dataframes to contain betas and predicted values
	lassoResults = pd.DataFrame(np.zeros((len(indNames), len(indNames))), columns = indNames, index = indNames)
	intercepts = pd.DataFrame(np.zeros(len(indNames)), columns = ["Intercept"])
	out = pd.DataFrame(np.zeros(len(indNames)), columns = ["yPred"], index = indNames)
	dfSliced = df.iloc[startRow:endRow + 1, 1:] # no date
	
	for indIndex in range(len(indNames)):
		X = dfSliced.iloc[:-1, :] 
		y = dfSliced.iloc[1:, indIndex] #ignore date column

		lasso = lassoM(X, y, method)
		xIndex = np.nonzero(lasso.coef_)[0]
		Xlin = X[X.columns[xIndex]]

		if xIndex != []:
			ols = linearM(Xlin, y)
			
			if mode == "predict":
				Xnew = [df.iloc[endRow + 1, xIndex + 1]] # shift selected predictors by 1 columns (date)
				out.iloc[indIndex, 0] = ols.predict(Xnew)[0]
			elif mode == "fit":
				j = 0
				intercepts.iloc[indIndex, 0] = ols.intercept_
				for i in xIndex:
					lassoResults.iloc[indIndex, i] = ols.coef_[j]
					j += 1

	#return intercepts, lassoResults
	return out if mode == "predict" else (intercepts, lassoResults)


def linRegression(df, method="aic", startRow = 0, endRow = 684, mode="fit"): #inclusive rows 
	indNames = list(df.iloc[:, 1:])
	#dataframes to contain betas and predicted values
	betas = pd.DataFrame(np.zeros((len(indNames), len(indNames))), columns = indNames, index = indNames)
	intercepts = pd.DataFrame(np.zeros(len(indNames)), columns = ["Intercept"])
	out = pd.DataFrame(np.zeros(len(indNames)), columns = ["yPred"], index = indNames)
	dfSliced = df.iloc[startRow:endRow + 1, 1:] # no date
	
	for indIndex in range(len(indNames)):
		X = dfSliced.iloc[:-1, :] 
		y = dfSliced.iloc[1:, indIndex] 


		ols = linearM(X, y)
		
		if mode == "predict":
			Xnew = [df.iloc[endRow + 1, 1:]] # shift selected predictors by 1 columns (date)
			out.iloc[indIndex, 0] = ols.predict(Xnew)[0]
		elif mode == "fit":
			j = 0
			intercepts.iloc[indIndex, 0] = ols.intercept_
			for i in range(len(indNames)):
				betas.iloc[indIndex, i] = ols.coef_[j]
				j += 1

	#return intercepts, betas
	return out if mode == "predict" else (intercepts, betas)


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