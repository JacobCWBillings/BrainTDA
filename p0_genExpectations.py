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
import argparse
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import squareform
from scipy.signal import butter, sosfiltfilt
from pycwt.helpers import fft, fft_kwargs, rect, ar1
from tqdm import tqdm
from pycwt.wavelet import wct_significance
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
 
    if args.nVols == -1:
        nVols = len(volunteers)
    else:
        nVols = args.nVols

    print('nVols is {}'.format(nVols))

    # Choose a spatial parcellation
    knn = p_utils.knn 
    lensPath = '/keilholz-lab/Jacob/TDABrains_00/data/groupICA_3T_HCP1200_MSMAll/groupICA/groupICA_3T_HCP1200_MSMAll_d{}.ica'.format(knn)
    lensName = 'melodic_IC.dscalar.nii'

    nICA = knn
    knn_dist_len = int(knn*(knn-1)/2)
    print(knn)

    buffLen = p_utils.buffLen
    mixLen = p_utils.mixLen
    trimLen = p_utils.trimLen
    trimEdges = p_utils.trimEdges
    
    # saveloc
    saveloc = './results/'    
    
    # Loop through volunteers. Does pipeline.    
    t0 = dt.datetime.now()        
    
    def doStack(voln):
        """Perform calculations of inter-volunteer dynamics"""
                
        curloc = saveloc + '/train0table/'
        makedirs(curloc ,exist_ok=True)

        curfile = (curloc + str(voln) + '.npz')
        #np.savez(curfile,'')

        adata = np.load(vloc + voln ).T               
        adata = np.delete(adata, p_utils.missingVoxels, axis=0)
 
        nvox, nT = nP, nT = np.shape(adata)
        print(nP,nT)
        print('Data single time histogram')
        print(np.histogram(adata[:,10].ravel()))
        
        rows, cols  = adata.shape
        ardata = []
        for row in range(rows):
            a1, _, _ = ar1(adata[row,:])
            ardata.append(a1)
                
        # Calculate significance threshold
        pool = Pool(args.poolSize)
        sj, frequencies, coi, mother, s0, dj, J, s1, TR = p_utils.getCWT_auxInfoPlus(adata[0,:])
       
        random.shuffle(ardata) 
        nar = len(ardata)
        nsamps = 25
        samps = np.arange(min(nar,nsamps))        
        print('sample ar coeffs histogram:')
        print(np.histogram(ardata[:nsamps]))
        mapSigs = partial(wct_significance,dt=p_utils.TR,dj=dj,s0=s0,J=J, significance_level=0.95, wavelet=mother,mc_count=40,progress=False,cache=False)
        sigs = list(pool.starmap(mapSigs, tqdm(list([ardata[sma],ardata[smb]] for sma,smb in itertools.combinations(samps,2)))))
    
        print(sigs[0])
    
        pool.close()
        pool.join()
        pool.terminate
      
        loc = saveloc + '/waveSigs/'
        makedirs(loc ,exist_ok=True)   
        name = (loc + str(voln) + '.pkl')
        with open(name, 'wb') as file:
            pickle.dump(sigs,file)

        return

    for voln in volunteers[:nVols]:
        if OSisfile(saveloc + '/train0table/' + str(voln) + '.npz'):          
            doStack(voln)
            pass
        else:
            doStack(voln)

if __name__ == '__main__':
    main()



