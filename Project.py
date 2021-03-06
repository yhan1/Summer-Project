import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LassoLarsIC, LinearRegression, LassoCV, LassoLarsCV
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
	#read in data
	indPath = "Data/30_Industry_Portfolios.CSV"
	rfPath = "Data/rf.csv"
	begDate = "195912"
	endDate = "201612"
	(ind, rf) = loadData(indPath, rfPath, begDate, endDate)
	exsReturns = excessReturns(ind, rf) 
	nrow = rf.count()[0]
	
	#create summary table (table 1 in paper)
	sumTable = summaryStat(exsReturns.iloc[:, 1:]) # no date

	#OLS post Lasso with entire time period
	indNames = list(exsReturns.iloc[:, 1:])
	df = exsReturns #with date
	(inter, fullPeriodResult)= OLSlassoRegression(df) # use aic
	
	# expanding period L/S portfolio construction
	startRow = 0
	endRow = df.loc[df["Date"] == 196912].index[0] #start before the first prediction date (paper uses 196912)
	lastRow = df.loc[df["Date"] == 201612].index[0]

	# start with first predicting row
	dateIndex = pd.period_range(start = pd.Period(str(df.loc[endRow + 1, "Date"]), freq = "M"), 
	                            end = pd.Period(str(df.loc[lastRow, "Date"]), freq = "M"), freq = "M") 

	# dataframes to contain return for each time period for ols post lasso, ols, and prevailing mean
	periodR = pd.DataFrame(np.zeros(lastRow - endRow), columns = ["r"], index = dateIndex)
	periodRLin = pd.DataFrame(np.zeros(lastRow - endRow), columns = ["r"], index = dateIndex)
	periodRPM = pd.DataFrame(np.zeros(lastRow - endRow), columns = ["r"], index = dateIndex)

	indBeta = pd.DataFrame(np.zeros((lastRow - endRow, len(indNames))), index = dateIndex, columns = indNames) 
	ind = 0 # food

	for e in range(endRow, lastRow):
	    #OLS POST LASSO OUTPUTS
	    (yPred, betas) = OLSlassoRegression(df, startRow = startRow, endRow = e, mode="predict")

	    # change not predicted returns to mean of cross section returns
	    yPred.loc[np.isnan(yPred)["yPred"], :] = np.nanmean(yPred, axis = 0)
	    yPred.sort_values(by = ["yPred"], ascending = True, inplace = True)
	    bottomInd = yPred.iloc[:6, :].index #find the industries
	    topInd = yPred.iloc[-6:, :].index
	    bottomR = df.loc[e + 1, bottomInd] #get the realized returns
	    topR = df.loc[e + 1, topInd]
	    
	    print(indBeta.index[e - endRow], np.round(np.average(topR), 2), np.round(np.average(bottomR), 2), np.round(np.average(topR) - np.average(bottomR), 2))
	    periodR.iloc[e - endRow, :] = np.mean(topR) - np.mean(bottomR)
	    indBeta.iloc[e - endRow, :] = betas.loc[betas.index[ind], :]
	    
	    #LINEAR MODEL OUTPUTS
	    (yPredLin, betasLin) = OLSRegression(df, startRow = startRow, endRow = e, mode="predict")
	    yPredLin.sort_values(by = ["yPred"], ascending = True, inplace = True)
	    bottomIndLin = yPredLin.iloc[:6, :].index #find the industries
	    topIndLin = yPredLin.iloc[-6:, :].index
	    bottomRLin = df.loc[e + 1, bottomIndLin] #get the realized returns
	    topRLin = df.loc[e + 1, topIndLin]
	    periodRLin.iloc[e - endRow, :] = np.mean(topRLin) - np.mean(bottomRLin)
	    
	    #PREVAILING MEAN OUPUTS
	    X = df.iloc[startRow:e + 1, 1:]
	    yPredPM = np.mean(X, axis = 0)
	    yPredPM.sort_values(ascending = True, inplace = True)
	    bottomIndPM = yPredPM[:6].index #find the industries
	    topIndPM = yPredLin.iloc[-6:, :].index
	    bottomRPM = df.loc[e + 1, bottomIndPM] #get the realized returns
	    topRPM = df.loc[e + 1, topIndPM]
	    periodRPM.iloc[e - endRow, :] = np.mean(topRPM) - np.mean(bottomRPM)



	writer = pd.ExcelWriter("portReturns.xlsx")
	periodR.to_excel(writer, "Sheet1")
	writer.save()

	writer = pd.ExcelWriter("OLSReturns.xlsx")
	periodRLin.to_excel(writer, "Sheet1")
	writer.save()

	writer = pd.ExcelWriter("PMReturns.xlsx")
	periodRPM.to_excel(writer, "Sheet1")
	writer.save()

	print(np.mean(periodR["r"]) * 12)




# plots betas over time 
def lineplot(x, y, xlab, ylab, title):
	xnew = range(len(x)) # equally spaced
	colors = cm.rainbow(np.linspace(0, 1, len(list(y))))
	for i in range(len(list(y))):
		if (len(np.nonzero(y.iloc[:, i])[0]) == 0): # industry not used at all
			plt.plot(xnew, y.iloc[:, i], label = "_nolegend_", color = colors[i])
		else:
			plt.plot(xnew, y.iloc[:, i], label = list(y)[i], color= colors[i])
	plt.legend(fontsize = "medium", loc=2)

	plt.title(title)
	plt.ylabel(ylab)
	plt.xlabel(xlab)
	plt.xticks(np.arange(len(x), step=12), x.asfreq("A").unique(), rotation = "vertical")

	plt.savefig(title)
	#plt.close()
	plt.show()


def OLSlassoRegression(df, method="aic", startRow = 0, endRow = 684, mode="fit"): #inclusive rows 
	indNames = list(df.iloc[:, 1:])
	#dataframes to contain betas and predicted values
	lassoResults = pd.DataFrame(np.zeros((len(indNames), len(indNames))), columns = indNames, index = indNames)
	intercepts = pd.DataFrame(np.zeros(len(indNames)), columns = ["Intercept"])
	out = pd.DataFrame(np.zeros(len(indNames)), columns = ["yPred"], index = indNames)
	dfSliced = df.iloc[startRow:endRow + 1, 1:] # no date
	
	for indIndex in range(len(indNames)):
		X = dfSliced.iloc[:-1, :] 
		y = dfSliced.iloc[1:, indIndex] 

		lasso = lassoM(X, y, method)
		xIndex = np.nonzero(lasso.coef_)[0]
		Xlin = X[X.columns[xIndex]]

		if xIndex != []:
			ols = linearM(Xlin, y)
			j = 0
			intercepts.iloc[indIndex, 0] = ols.intercept_
			
			for i in xIndex:
				lassoResults.iloc[indIndex, i] = ols.coef_[j]
				j += 1
			if mode == "predict":
				Xnew = [df.iloc[endRow + 1, xIndex + 1]] # shift selected predictors by 1 columns (date)
				out.iloc[indIndex, 0] = ols.predict(Xnew)[0]
	
		else: # no x variables selected, then best prediction of y is average of y
			# intercepts.iloc[indIndex, 0] = np.average(y)
			# out.iloc[indIndex, 0] = np.average(y)
			out.iloc[indIndex, 0] = np.nan
				

	#return intercepts, lassoResults
	return (out, lassoResults) if mode == "predict" else (intercepts, lassoResults)


def OLSRegression(df, method="aic", startRow = 0, endRow = 684, mode="fit"): #inclusive rows 
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
		j = 0
		intercepts.iloc[indIndex, 0] = ols.intercept_
		for i in range(len(indNames)):
			betas.iloc[indIndex, i] = ols.coef_[j]
			j += 1
		if mode == "predict":
			Xnew = [df.iloc[endRow + 1, 1:]] # shift selected predictors by 1 columns (date)
			out.iloc[indIndex, 0] = ols.predict(Xnew)[0]

	#return intercepts, betas
	return (out, betas) if mode == "predict" else (intercepts, betas)


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