########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/11/14
########################################################

import numpy as np 
from numpy import linalg as LA
import time
import random
import core
from utilities import *


########################################################
# Function to run the prediction approach at each density
# 
def execute(matrix, density, para, sliceId):
    startTime = time.clock()
    numService = matrix.shape[1] 
    numUser = matrix.shape[0] 
    rounds = para['rounds']
    logger.info('Data matrix size: %d users * %d services'%(numUser, numService))
    logger.info('Run the algorithm for %d rounds: matrix density = %.2f.'%(rounds, density))
    evalResults = np.zeros((rounds, len(para['metrics']))) 
    timeResults = np.zeros((rounds, 1))
    	
    for k in range(rounds):
		logger.info('----------------------------------------------')
		logger.info('%d-round starts.'%(k + 1))
		logger.info('----------------------------------------------')

		# remove the entries of data matrix to generate trainMatrix and testMatrix
		seedID = k + sliceId * 100
		(trainMatrix, testMatrix) = removeEntries(matrix, density, seedID) 
		logger.info('Removing data entries done.')
		(testVecX, testVecY) = np.where(testMatrix)		
		testVec = testMatrix[testVecX, testVecY]
		# read the training data, i.e., removed matrix

		# invocation to the prediction function
		iterStartTime = time.clock() # to record the running time for one round             
		predictedMatrix = core.predict(trainMatrix, para) 		
		timeResults[k] = time.clock() - iterStartTime

		# calculate the prediction error
		predVec = predictedMatrix[testVecX, testVecY]
		evalResults[k, :] = errMetric(testVec, predVec, para['metrics'])

		logger.info('%d-round done. Running time: %.2f sec'%(k + 1, timeResults[k]))
		logger.info('----------------------------------------------')

    outFile = '%s%02d_%sResult_%.2f.txt'%(para['outPath'], sliceId + 1, para['dataType'], density)
    saveResult(outFile, evalResults, timeResults, para)
    logger.info('Config density = %.2f done. Running time: %.2f sec'
			%(density, time.clock() - startTime))
    logger.info('==============================================')
########################################################


########################################################
# Function to remove the entries of data matrix
# Use guassian random sampling
# Return trainMatrix and testMatrix
#
def removeEntries(matrix, density, seedID):
	numAll = matrix.size
	numTrain = int(numAll * density)
	(vecX, vecY) = np.where(matrix > -1000)
	np.random.seed(seedID % 100)
	randPermut = np.random.permutation(numAll)	
	np.random.seed(seedID)
	randSequence = np.random.normal(0, numAll / 6.0, numAll * 10)

	trainSet = []
	flags = np.zeros(numAll)
	for i in xrange(randSequence.shape[0]):
		sample = int(abs(randSequence[i]))
		if sample < numAll:
			idx = randPermut[sample]
			if flags[idx] == 0 and matrix[vecX[idx], vecY[idx]] > 0:
				trainSet.append(idx)
				flags[idx] = 1
		if len(trainSet) == numTrain:
			break
	if len(trainSet) < numTrain:
		logger.critical('Exit unexpectedly: not enough data for density = %.2f.', density)
		sys.exit()

	trainMatrix = np.zeros(matrix.shape)
	trainMatrix[vecX[trainSet], vecY[trainSet]] = matrix[vecX[trainSet], vecY[trainSet]]
	testMatrix = np.zeros(matrix.shape)
	testMatrix[matrix > 0] = matrix[matrix > 0]
	testMatrix[vecX[trainSet], vecY[trainSet]] = 0

    # ignore invalid testing users or services             
	idxX = (np.sum(trainMatrix, axis=1) == 0)
	testMatrix[idxX, :] = 0
	idxY = (np.sum(trainMatrix, axis=0) == 0)
	testMatrix[:, idxY] = 0    
	return trainMatrix, testMatrix
########################################################


########################################################
# Function to compute the evaluation metrics
#
def errMetric(realVec, predVec, metrics):
    result = []
    absError = np.abs(predVec - realVec) 
    mae = np.sum(absError)/absError.shape
    for metric in metrics:
	    if 'MAE' == metric:
			result = np.append(result, mae)
	    if 'NMAE' == metric:
		    nmae = mae / (np.sum(realVec) / absError.shape)
		    result = np.append(result, nmae)
	    if 'RMSE' == metric:
		    rmse = LA.norm(absError) / np.sqrt(absError.shape)
		    result = np.append(result, rmse)
	    if 'MRE' == metric or 'NPRE' == metric:
	        relativeError = absError / realVec
	        relativeError = np.sort(relativeError)
	        if 'MRE' == metric:
		    	mre = np.median(relativeError)
		    	result = np.append(result, mre)
	        if 'NPRE' == metric:
		    	npre = relativeError[np.floor(0.9 * relativeError.shape[0])] 
		    	result = np.append(result, npre)
    return result
########################################################