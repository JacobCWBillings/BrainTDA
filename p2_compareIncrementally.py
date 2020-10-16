"""Calculates BOLD dynamics metric space, per volunteer.

Argument list: 
apath: Reads list of volunteers from here 
       (default: '/keilholz-lab/SharedFiles/SomeBrainMaps/HCPSaves_Compact/').
poolsize: Size of cpu pool (default: 14).
nVols: Number of volunteer datasets to run (default: -1 "all").
display: plot and display some intermediate results (default: False).
"""

from os.path import join as OSjoin
from os.path import isfile as OSisfile
from os import makedirs as makedirs
from multiprocessing import Pool, cpu_count, Process, Event
from filelock import SoftFileLock as sfl
import argparse
import random
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import squareform
from scipy.signal import butter, sosfiltfilt
from pycwt.helpers import fft, fft_kwargs, rect
from tqdm import tqdm
import datetime as dt
import itertools
from functools import partial

import p_utils
import plotters

maxdim = p_utils.maxdim
truncDim = p_utils.truncDim
print([truncDim,maxdim])
TR = p_utils.TR

TARGS = p_utils.TARGS
TARGS_CEIL = p_utils.TARGS_CEIL
frq_edges = p_utils.frq_edges

def main(event):    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--apath', type=str, default = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/anat/')
    parser.add_argument('--poolSize', type=int, default=14)
    parser.add_argument('--nVols', type=int, default=-1)
    parser.add_argument('--display', type=bool, default=False)
    
    args = parser.parse_args()    
    
    apath_rest = args.apath
    apath_task = args.apath
    
    vloc = args.apath   
    volunteers, restDirs, taskDirs, EVtxt, Blocks, Blockstxt = p_utils.getLists(vloc=vloc)
    random.shuffle(volunteers)

    # saveloc
    saveloc = './results/'

    poolSize = args.poolSize    
    
    maxdim = p_utils.maxdim
    metrics = p_utils.metrics

    train_volloc = saveloc + 'ind0/'    
        
    # Make table of existing data
    # indices pulled from train_volloc as previous step (p3_....py) runs over all available volunteers
    all_times_vols = p_utils.getSecondList(train_volloc)
    allInds = []
    volInds = {}
    allNames = []
    allNames_dict = {}
    dict_count = -1
    dLen = 0
    for voln in tqdm(all_times_vols,desc='Loading timing info.'):
        allInds.append(np.load(train_volloc + str(voln) + '.npy', allow_pickle=True))
        volInds[voln] = np.empty(allInds[-1].shape,dtype = 'int')
        for i in allInds[-1]:            
            dict_count += 1
            allNames.append( (voln[:5] + '_{:03d}').format(i) )    
            allNames_dict[allNames[-1]] = dict_count
            volInds[voln][i] = dict_count            
    dLen = i+1
    lan = len(allNames)
    san = set(allNames)
            
    # Also load data                
    UUs = {}                  
    for metric in metrics:
        UUs[metric] = []
        for voln in tqdm(all_times_vols, desc='Loading data'):
            if metric=='diagram':
                graphType = 'ripped'            
            elif metric=='simplex':
                graphType = 'simplex'            
            locU = saveloc + '{}Training/'.format(graphType)        
            with open(locU + str(voln) + '.pkl', 'rb') as file:
                uus = pickle.load(file)        
            for uu in uus:
                if metric == 'diagram':
                    UUs[metric].append(uu['dgms'])
                if metric == 'simplex':
                    UUs[metric].append(uu)
    print('UUs[-1] is shape {}'.format(UUs[metric][-1].shape))
    nP = p_utils.knn
    nW = 1
    pool = Pool(poolSize, p_utils.getMultiscaleIndexerRanges, (nP, nW, ))
            
    # Initialize data Mat            
    dataMat = {}
    for metric in metrics:
        for dim in range(maxdim+1):
            dataMat[dim] = np.full([lan,lan],-1.0)
            dataMat[dim][np.diag_indices(lan)] = 0

    # Load finished distances
    for metric in metrics:        
        
        if metric == 'diagram':
            dist_fun = partial(p_utils.slw2, normalize=False)
            desc = 'calc Wasserstein distance'
        elif metric == 'simplex':
            dist_fun = partial(p_utils.calcWeightedJaccard)
            desc = 'calc connected simplices'
            
        locBeg = saveloc + '{}AllDist_HN/Begun/'.format(metric)    
        makedirs(locBeg ,exist_ok=True)
        begun = p_utils.getSecondList(locBeg)            
        
        locFin = saveloc + '{}AllDist_HN/Finished/'.format(metric)    
        makedirs(locFin ,exist_ok=True)
        finished = p_utils.getSecondList(locFin)                            
        for fin in tqdm(finished):
            fname = locBeg + fin + '.npy'  
            otherInds = np.load(fname)
            fname = locFin + fin + '.npy'
            tempD = np.load(fname)
            for dim in range(maxdim+1):            
                dataMat[dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                dataMat[dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()                
                    
        remaining = san-set(finished)-set(begun)
        
        while len(remaining) and not event.is_set():
            print('finished {}'.format(-len(remaining) + lan))                  
            print('remaining Vol x Times, {} of {}.'.format(len(remaining), lan) )
            
            lineLbl = random.choice(tuple(remaining))
            lineNum = allNames_dict[lineLbl]
            otherInds = np.arange(lan)[(dataMat[0][lineNum,:]==-1).ravel()]
            
            np.save(locBeg + lineLbl + '.npy', otherInds)
            
            #print('printing dataMat histograms')
            #for dim in range(maxdim+1):            
            #    print(np.histogram(dataMat[dim].ravel()))
                                   
            print('For datapoint {}, # remaining inds = {}.'.format(lineLbl,len(otherInds)))

            
            tempD = list(pool.imap(dist_fun, tqdm([ [UUs[metric][lineNum], UUs[metric][oi]]
                               for oi in otherInds ], total=len(otherInds), desc = desc),
                                   chunksize=1) )            
            #tempD = [[-1]*len(Inds)]*3
            tempD = np.array(tempD)

            print('Saving results for line {}'.format(lineLbl))                    
            fname = locFin + lineLbl + '.npy'            
            np.save(fname,tempD)
     
            newly_finished = p_utils.getSecondList(locFin)                            
            
            for fin in list(set(newly_finished) - set(finished)):
                fname = locBeg + fin + '.npy'
                otherInds = np.load(fname)
                fname = locFin + fin + '.npy'
                tempD = np.load(fname)
                for dim in range(maxdim+1):            
                    dataMat[dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                    dataMat[dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()
                    
            finished = newly_finished                    
            begun = p_utils.getSecondList(locBeg)                        
            remaining = san-set(finished)-set(begun)                 
            
        print('finished {}'.format(-len(remaining) + lan))                  
        print('We done with {}'.format(metric))
                    
    return

if __name__ == '__main__':
    event = Event()
    p = Process(target=main, args=(event,) )
    p.start()
    
    userInput = None
    while userInput != 'quit':
        userInput = str(input('Type ''quit'' to stop processing'))
        
    print('*********Initiating shutdown***********')
    event.set()
        
    p.join()
    p.terminate()







