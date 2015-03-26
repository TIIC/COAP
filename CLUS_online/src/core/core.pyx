########################################################
# core.pyx
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/10/22
########################################################

import time
import numpy as np
from utilities import *
cimport numpy as np # import C-API
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport *
import scipy.cluster # import K-means


#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "CLUS.h":
    void CLUS(double *removedData, double *predData, int numUser, 
        int numService, int numTimeSlice, vector[int] attrEv, 
        vector[int] attrUs, vector[int] attrWs, vector[vector[int]] clusterEv, 
        vector[vector[int]] clusterUs, vector[vector[int]] clusterWs, 
        bool debugMode)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(trainTensor, para):
    # obtain training data
    if para['dataType'] == 'rt':
        startSlice = int(trainTensor.shape[2] * (1 - para['slicesToTest']))
    else: # for case 'rel'
        startSlice = 0
    offlineTrainTensor = trainTensor[:, :, 0:startSlice]
    (numUser, numService, numTimeSlice) = offlineTrainTensor.shape
    numEvClus = para['numEvClus']
    numUsClus = para['numUsClus']
    numWsClus = para['numWsClus']

    vecEv = np.zeros(numTimeSlice)
    vecUs = np.zeros((numUser, numEvClus))
    vecWs = np.zeros((numService, numEvClus))
    clusterEv = [[] for i in xrange(numEvClus)]
    clusterUs = [[] for i in xrange(numUsClus)]
    clusterWs = [[] for i in xrange(numWsClus)]

    # time tracking
    startTime = time.clock()
    
    # clustering
    for i in xrange(numTimeSlice):
        vecEv[i] = np.sum(offlineTrainTensor[:, :, i]) / (np.sum(offlineTrainTensor[:, :, i] > 0) + np.spacing(1))
    [evCentroid, attrEv] = scipy.cluster.vq.kmeans2(np.matrix(vecEv).T, numEvClus, minit = 'points')
    
    for i in xrange(numTimeSlice):
        clusterEv[attrEv[i]].append(i)
    logger.info('Time side clustering done.')

    for i in xrange(numUser):
        for j in xrange(numEvClus):
            vecUs[i, j] = np.sum(offlineTrainTensor[i, :, clusterEv[j]])\
            / (np.sum(offlineTrainTensor[i, :, clusterEv[j]] > 0) + np.spacing(1))
    [_, attrUs] = scipy.cluster.vq.kmeans2(vecUs, numUsClus, minit = 'points')
    for i in xrange(numUser):
        clusterUs[attrUs[i]].append(i)
    logger.info('User side clustering done.')
        
    for i in xrange(numService):
        for j in xrange(numEvClus):
            vecWs[i, j] = np.sum(offlineTrainTensor[:, i, clusterEv[j]])\
            / (np.sum(offlineTrainTensor[:, i, clusterEv[j]] > 0) + np.spacing(1))
    [_, attrWs] = scipy.cluster.vq.kmeans2(vecWs, numWsClus, minit = 'points')
    for i in xrange(numService):
        clusterWs[attrWs[i]].append(i)
    logger.info("Service side clustering done.")

    # initialization
    cdef np.ndarray[double, ndim=3, mode='c'] predEvTensor =\
        np.zeros((numUser, numService, numEvClus))
    cdef vector[vector[int]] vecClusterEv = clusterEv
    cdef vector[vector[int]] vecClusterUs = clusterUs
    cdef vector[vector[int]] vecClusterWs = clusterWs
    cdef vector[int] vecAttrEv = attrEv
    cdef vector[int] vecAttrUs = attrUs
    cdef vector[int] vecAttrWs = attrWs
    cdef bool debugMode = para['debugMode']

    # wrap up CLUS.cpp
    CLUS(
        <double *> (<np.ndarray[double, ndim=3, mode='c']> offlineTrainTensor.copy()).data,
        <double *> predEvTensor.data,
        <int> numUser,
        <int> numService,
        <int> numTimeSlice,
        vecAttrEv,
        vecAttrUs,
        vecAttrWs,
        vecClusterEv,
        vecClusterUs,
        vecClusterWs,
        debugMode  
        )

    # time tracking
    trainDoneTime = time.clock()
    trainingTime = trainDoneTime - startTime

    # online prediction via CLUS model
    onlineTrainTensor = trainTensor[:, :, startSlice:]
    predTensor = np.zeros(onlineTrainTensor.shape)
    for i in xrange(onlineTrainTensor.shape[2]):
        onlineFeature = np.sum(onlineTrainTensor[:, :, i]) /\
            (np.sum(onlineTrainTensor[:, :, i] > 0) + np.spacing(1))
        distVec = np.absolute(onlineFeature - evCentroid)
        clusterIdx = np.argmin(distVec)
        predMatrix = predEvTensor[:, :, clusterIdx]
        # fill with overall average value for entries without predictions
        predMatrix[predMatrix <= 0] = onlineFeature
        predTensor[:, :, i] = predMatrix
    predTensor[onlineTrainTensor > 0] = onlineTrainTensor[onlineTrainTensor > 0]

    # time tracking
    predictingTime = time.clock() - startTime
    runningTime = [trainingTime, predictingTime / onlineTrainTensor.shape[2]]
    
    return predTensor, runningTime
#########################################################  

