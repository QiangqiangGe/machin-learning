## adaboost 
##加载数据集
import numpy as np 
def loadSimpData():
	datMat = matrix([[1,2.1],
		[2,1.1],
		[1.3,1],
		[1,1],
		[2,1]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

##单层决策树生成函数
###通过阈值对数据进行分类，在阈值一侧的为-1，另一侧为+1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = np.ones((np.shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:dimen] > threshVal] = -1.0
    return retArray
##建立决策树
def buildStump(dataArr,classLabels,D):
	dataMatrix = np.mat(dataArr)
	labelMat = np.mat(classLabels).T 
	m,n = np.shape(dataMatrix) 
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	minError = float("inf")
	##两列
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax-rangeMin)/numSteps
		##步长
		for j in range(-1,int(numSteps)+1):
			##大于或小于当前阈值
			for inequal in ['lt','gt']:
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,threshVal,inequal)
				errArr = np.mat(np.ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T * errArr
				print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"%(i,threshVal,ineqal,weightedError)
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump，minError,bestClasEst

##构建AdaBoost训练过程
def adaBoostTrainDs(dataArr,classLabels,numIt=40):
	weakClassArr = []
	m = np.shape(dataArr)[0]
	D = np.mat(np.ones((m,1)))
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		print "D:",D.T
		alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		print "classEst: ",classEst.T
		expon = np.multipy(-1*alpha*np.mat(classLabels).T,classEst)
		D = np.multipy(D,np.exp(expon))
		D = D/D.sum()
		aggClassEst += alpha * classEst
		print "aggClassEst: ",aggClassEst.T 
		aggErrors = np.multipy(sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
		errorRate = aggErrors.sum()/m
		print "total error: ",errorRate,"\n"
		if errorRate == 0.0: break
	return weakClassArr
