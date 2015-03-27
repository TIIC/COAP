########################################################
# run_rt.py: response-time prediction
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2015/03/12
# Implemented approach: CLUS
# Evaluation metrics: MAE, NMAE, RMSE, MRE, NPRE
########################################################

import numpy as np
import os, sys, time
import multiprocessing
sys.path.append('src')
# Build external model
if not os.path.isfile('src/core.so'):
	print 'Lack of core.so (built from the C++ module).' 
	print 'Please first build the C++ code into core.so by using: '
	print '>> python setup.py build_ext --inplace'
	sys.exit()
from utilities import *
import evaluator
import dataloader
 

#########################################################
# config area
#
para = {'dataType': 'rt', # choose 'rt' for response-time prediction
		'dataPath': '../data/rt_data/rtdata.txt',
		'outPath': 'result/',
		'slicesToTest': 0.1, # use the last 10% of the time slices as testing data
		'metrics': ['MAE', 'NMAE', 'RMSE'], # delete where appropriate		
		'density': list(np.arange(0.02, 0.11, 0.02)), # matrix density
		'rounds': 20, # how many runs are performed at each matrix density
        'numEvClus': 10, # number of clusters of environment
        'numUsClus': 10, # number of clusters of users
        'numWsClus': 10, # number of clusters of services
		'saveTimeInfo': False, # whether to keep track of the running time
		'saveLog': True, # whether to save log into file
		'debugMode': False, # whether to record the debug info
        'parallelMode': True # whether to leverage multiprocessing for speedup
		}

initConfig(para)
#########################################################


startTime = time.clock() # start timing
logger.info('==============================================')
logger.info('CLUS: [Silic et al., FSE\'2013].')

# load the dataset
dataTensor = dataloader.load(para)
logger.info('Loading data done.')

# run for each density
if para['parallelMode']: # run on multiple processes
    pool = multiprocessing.Pool()
    for density in para['density']:
        pool.apply_async(evaluator.execute, (dataTensor, density, para))
    pool.close()
    pool.join()
else: # run on single processes
    for density in para['density']:
		evaluator.execute(dataTensor, density, para)

logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
         time.gmtime(time.clock() - startTime)))
logger.info('==============================================')
sys.path.remove('src')