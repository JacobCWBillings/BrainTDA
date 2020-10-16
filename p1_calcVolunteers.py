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
    
    # saveloc
    saveloc = './results/'    
    
    # Loop through volunteers. Does pipeline.    
    t0 = dt.datetime.now()        
    
    def doStack(voln):
        """Perform calculations of inter-volunteer dynamics"""
                
        curloc = saveloc + '/train0table/'
        makedirs(curloc ,exist_ok=True)

        curfile = (curloc + str(voln) + '.npz')
        np.savez(curfile,'')

        adata = np.load(vloc + voln ).T        
       
        adata = np.delete(adata, p_utils.missingVoxels, axis=0)
 
        nvox, nT = nP, nT = np.shape(adata)
        print(nP,nT)
        print('Data single time histogram')
        print(np.histogram(adata[:,10].ravel()))
        '''
        testData = np.sum(adata,axis=1)
        if sum(testData==0)>0:
            print('skipping volunteer {} with {} zero valued parcels'.format(voln, sum(testData==0)))
            return              
        ''' 
        sj, frequencies, coi, mother, s0, dj, J, s1, TR = p_utils.getCWT_auxInfoPlus(adata[0,:])

        pool = Pool(args.poolSize)
        
        '''
        W = pool.map(p_utils.getCWT_coeff,tqdm(adata))        

        W = np.array(W)
        #W = W[:10,:]
        print('shape W is ' + str(W.shape))
        print('W histogram')
        print(np.histogram(W.ravel()))
                
        _, m, n = nP, nW, nT = W.shape 
                       
        k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, 0, :])['n'])
        k2 = k ** 2            
        scales1 = np.ones([1, n]) * sj[:, None]
        snorm = sj / TR
        F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
        wsize = mother.deltaj0 / dj * 2
        win = rect(np.int(np.round(wsize)), normalize=True)                                                
       
        print('Smooth wavelets')
        mapFun = partial(p_utils.cwt2smooth_local, n=n, F=F, win=win)
        S = pool.map(mapFun,tqdm( list(np.abs(W[ic,:,:].squeeze())**2/scales1 for ic in range(nP)) ))                
        print('One smoothed values histogram')
        print(np.histogram(S[0].ravel()))

        samps = np.arange(nP)
              
        mapCross = partial(p_utils.cross)
        W12 = list(pool.imap(mapCross,tqdm(list([W[sma,:,:].squeeze(),W[smb,:,:].squeeze()] for sma, smb in itertools.combinations(samps,2)), desc='Cross wavelets')))
        mapCoher = partial(p_utils.coher_postCross,n=n,F=F,win=win,scales1=scales1)
        wct = list(pool.imap(mapCoher,tqdm(list([W12[i],S[sma],S[smb]] for i,[sma,smb] in enumerate(itertools.combinations(samps,2))), desc='Wavelet coherence' )))
        mapNorms = partial(p_utils.crossNorms, frq_edges=frq_edges)
        nW12 = list(pool.imap(mapNorms,tqdm(W12)))
        print('one W12 norms histogram, having shape {}'.format(nW12[0].shape))
        print(np.histogram(nW12[0]))

        pool.close()
        pool.join()
        pool.terminate

        # Trim input data
        temp = np.array(wct)
        temp = temp[:,:,trimLen:-trimLen]
        tnW12 = np.array(nW12)[:,:,trimLen:-trimLen]
        aa, bb, cc = temp.shape
        wct = np.zeros((aa, 1, cc))
        [uu0, uu1] = frq_edges
        wct[:,0,:] = np.sum(np.multiply(temp[:,uu0:uu1,:], tnW12[:,uu0:uu1,:]),axis=1)
        _, m, n = _, nW, nT = wct.shape
        print('wct is shape {}.'.format(wct.shape))
        print('One wct histogram')
        print(np.histogram(wct[0].ravel()))
        
        #Select time points, random
        #inds = p_utils.doAgg(voln, adf)
        inds = np.arange(nT) #np.random.randint(nT,size=75)
        nI = len(inds)

        indloc = saveloc + '/ind0/'
        makedirs(indloc ,exist_ok=True)
        indfile = (indloc + str(voln) + '.npy')
        np.save(indfile,inds)
        
        wctloc = saveloc + '/wctTraining/'
        makedirs(wctloc ,exist_ok=True)
        wctfile = (wctloc + str(voln) + '.pkl')
        with open(wctfile, 'wb') as file:
            pickle.dump(wct[:,:,inds],file, pickle.HIGHEST_PROTOCOL)
        '''

        wctfile = ('../z18_GonzCast_WeightedCWT/results/wctTesting/' + str(voln) + '.pkl')
        with open(wctfile, 'rb') as file:
            wct = pickle.load(file)
            
        _, m, n = wct.shape
        nP = p_utils.knn
        nW = 1
        nI = n
        inds = np.arange(nI)
            
        # Build background connectivity
        print('Building background connectivity')
        Wcoh1 = np.ones((nP*nW,nP*nW)) # setting background to infinate (=1) distance
        Wcoh0 = np.zeros((nP*nW,nP*nW))
        Wcoh = [1]*nI
        GG = {}
        inc = 0
        for ee, ii in enumerate(tqdm(inds)):
            temp = Wcoh0.copy()
            inc = 0
            for nw in range(nW):
                temp[inc:inc+nP:,inc:inc+nP:] = 1-squareform(wct[:,nw,ii].ravel())
                inc += nP
            rows, cols = np.nonzero(temp)
            temp[np.diag_indices(nP*nW)] = 0
            Wcoh[ee] = Wcoh1.copy()
            Wcoh[ee][rows,cols] = temp[rows,cols]

        print('Wcoh is len {}'.format(len(Wcoh)))
        '''
        pool = Pool(min(nI,args.poolSize), p_utils.getMultiscaleIndexer, (nP, nW, ))
        
        bd_fun = partial(p_utils.calcSimplices)
        bdMat = list(pool.imap(bd_fun, tqdm([[tt, Wcoh[tt]] for tt in range(nI)], desc='calc connected simplices'), chunksize=1)) 
        #bdMat = list(map(bd_fun, tqdm([[tt, Wcoh[tt]] for tt in range(nI)], desc='calc connected simplices')))            

        loc = saveloc + '/simplexTraining/'
        makedirs(loc ,exist_ok=True)   
        name = (loc + str(voln) + '.pkl')
        with open(name, 'wb') as file:
            pickle.dump(bdMat,file)

        '''
        pool = Pool(min(nI,args.poolSize))
        ripFun = partial(p_utils.getRipped, 
                         maxdim=1, 
                         makeSquare = False, 
                         fixInf = True, 
                         do_cocycles=True, 
                         threshold=p_utils.TARGS_CEIL,
                         doLandscape=False)
        Rip = list(pool.imap(ripFun, tqdm([Wcoh[tt] for tt in range(nI)], desc='Rips fun'), chunksize=1))

        riploc = saveloc + '/rippedTraining/'
        makedirs(riploc ,exist_ok=True)

        ripfile = (riploc + str(voln) + '.pkl')
        with open(ripfile, 'wb') as file:
            pickle.dump(Rip,file, pickle.HIGHEST_PROTOCOL)

        for r in range(maxdim+1):
            print('One lifetimes histogram in dimension {}'.format(r))
            print(np.histogram(np.diff(Rip[0]['dgms'][r],axis=1)))
        

        pool.close()
        pool.join()
        pool.terminate
                
        #if args.display:
        #    padmFun = partial(plotters.plotCoherMetrics, display=args.display)
        #    Process(target=padmFun)
        return

    for voln in volunteers[:nVols]:
        if OSisfile(saveloc + '/train0table/' + str(voln) + '.npz'):          
            pass
        else:
            doStack(voln)   

if __name__ == '__main__':
    main()


