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
from multiprocessing import Pool, cpu_count, Process
from filelock import SoftFileLock as sfl
import argparse
import random
import pickle

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
import umap

import p_utils
import plotters

maxdim = p_utils.maxdim
truncDim = p_utils.truncDim
print([truncDim,maxdim])
TR = p_utils.TR

TARGS = p_utils.TARGS
TARGS_CEIL = p_utils.TARGS_CEIL
frq_edges = p_utils.frq_edges

def main():    
    
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
    pool = Pool(poolSize)
    
    maxdim = p_utils.maxdim
    metrics = p_utils.metrics

    train_volloc = saveloc + 'ind0/'    
    test_volloc = saveloc + 'ind1/'
        
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
            
    # Initialize data Mat            
    dataMat = {}
    for metric in metrics:
        print('Current metric is {}'.format(metric))
        for dim in range(maxdim+1):
            dataMat[dim] = np.full([lan,lan],1000.0)
            dataMat[dim][np.diag_indices(lan)] = 0

    # Load distances
    for metric in metrics:        
        locBeg = saveloc + '{}AllDist_HN/Begun/'.format(metric)    
        makedirs(locBeg ,exist_ok=True)
        begun = p_utils.getSecondList(locBeg)            
        
        locFin = saveloc + '{}AllDist_HN/Finished/'.format(metric)    
        makedirs(locFin ,exist_ok=True)
        finished = p_utils.getSecondList(locFin)                            
        for fin in tqdm(finished,desc='Load data'):
            fname = locFin + fin + '.npy'
            tempD = np.load(fname)
            fname = locBeg + fin + '.npy'
            otherInds = np.load(fname)
            for dim in range(maxdim+1):            
                dataMat[dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                dataMat[dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()                                            

    allFin = list(allNames_dict[fin] for fin in finished)    
    # trim finished
    allFin = np.array(list(allNames_dict[fin] for fin in finished))
    T = volInds[all_times_vols[0]][-1]+1
    # trim finished
    someFin = []
    #args.nVols = 15
    if args.nVols > -1:
        print('Trimming allFin from len {}'.format(len(allFin)))
        for vi, voln in tqdm(enumerate(all_times_vols), desc='Trimming group embedding inputs.'):  
            bounds = (vi*T, (vi+1)*T-1)
            vec = allFin[(allFin>=bounds[0]) * (allFin<=bounds[1])]
            while len(vec)>args.nVols:
                vecd = np.diff(vec)
                vecs = np.sum(np.concatenate([vecd[:-1][:,None],vecd[1:][:,None]],axis=1),axis=1)
                pop = np.argmin(vecs)
                vec = np.delete(vec,pop+1)
            someFin.extend(vec)
        allFin = someFin        
        print('allFin trimmed to len {}'.format(len(allFin)))
    
    for metric in metrics: 
        print('Make group embedding')
        embedG = {}        
        for dim in range(maxdim+1):        
            
            groupG = dataMat[dim][allFin,:][:,allFin]
            reducer = umap.UMAP(n_neighbors=300, n_components=2, metric=p_utils.donotripit, n_epochs=None, learning_rate=1.0, init='random', min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=2.0, repulsion_strength=5.0, negative_sample_rate=2, transform_queue_size=16.0, a=None, b=None, random_state=None, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, verbose=True)
            embedG[dim] = reducer.fit(groupG)
            print(np.histogram(embedG[dim].embedding_))
        
        print(len(embedG))
        for voln in all_times_vols:          
            print('Make {} embedding'.format(voln))
            Dembs = {}
            for dim in range(maxdim+1):        
            
                volFin = volInds[voln]
                group = dataMat[dim][volFin,:][:,allFin]
                print(group.shape)
                Dembs[dim] = embedG[dim].transform(group)
                  
                print(np.histogram(Dembs[dim]))                
                
            if metric == 'diagram':    
                loc = saveloc + 'UMAPxyAllDiagram/'
            elif metric == 'simplex':
                loc = saveloc + 'UMAPxyAllSimplex/'
            elif metric == 'strength':
                loc = saveloc + 'UMAPxyAllStrength/'
            print(metric)
            print(loc)
            makedirs(loc ,exist_ok=True)
            file = (loc + str(voln) + '.pkl')
            with open(file, 'wb') as sfile:
                pickle.dump(Dembs,sfile, pickle.HIGHEST_PROTOCOL)        
    
    return
          
if __name__ == '__main__':
    main()







