import pickle
from os import makedirs as makedirs

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import squareform, cdist, pdist
from functools import partial
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import watershed
import shapely.geometry as geo
from shapely.ops import snap
import scipy.stats as st
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm import tqdm
from pyitlib import discrete_random_variable as drv

from p_utils import *
import p_utils

# Static parameters

curdir = './results/'

buffLen = p_utils.buffLen
truncDim = p_utils.truncDim	
trimLen = p_utils.trimLen

goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']

# General method for plotting graphs
def plotMatrixValues(G,fname='GraphValues.png',save=True,display=False):
    fig = plt.figure(num=1,figsize=[8.5,11])
    
    ax = plt.subplot(111)
    img = plt.imshow(G)

    if save:
        plt.savefig(fname)

    if display:
        plt.show()
    plt.close()

def dispCoherHist(saveloc = curdir, nVol = -1):
    
    wavloc = saveloc + 'waveSigs/'
    
    volunteers = getSecondList(wavloc)

    wav = []
    
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:            
            wavfile = wavloc + str(vol) + '.pkl'    
            with open(wavfile,'rb') as file:
                wav.append(pickle.load(file))                        

    gg = np.concatenate(wav)
    print('Significance wave coher, mean. nVol = {}'.format(vi+1))
    print(np.nanmean(gg,axis=0))
    print('Significance wave coher, std. nVol = {}'.format(vi+1))
    print(np.nanstd(gg,axis=0))
        
def plotCoherMetrics(saveloc = curdir, display=False, nVol = -1):
    
    tabloc = saveloc + 'train0table/'
    wavloc = saveloc + 'wcoherenceFiles/'

    volunteers = getSecondList(wavloc)

    ADF = []
    wav = []
    
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            curtab = tabloc + str(vol) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF.append(pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times']))
        
            wavfile = wavloc + str(vol) + '.pkl'    
            with open(wavfile,'rb') as file:
                wav.append(pickle.load(file))
            
            if vi == 0:
                print('adf is shape {}, wav is len {}'.format(ADF[-1].shape,len(wav[-1])))

        else:
            pass

    Nv = nV = len(wav)

    trimLen = int(350)
    goodStates = ['mental','rnd','2bk_body','2bk_faces','2bk_places','2bk_tools']
    adf = []    
    ce = []
    ct = []
    
    for ni in range(Nv):
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values             
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        aset = [adf[ni].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        ce.append(np.argmax(aset, axis=0))
        ct.append(np.max(aset, axis=0))                         

        if ni == 0:
            print('ce histogram : {}'.format(np.histogram(ce)))
            
    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   
    
    for ii, st in enumerate(goodStates):
        fig = plt.figure(1,figsize=[18,10])
        name = (loc + st + '.png')            
        state_block = wav[0][0].todense()*0
        countt = 0
        for nv in range(Nv):
            for jj, wv in enumerate(wav[nv]):
                if ce[nv][jj]==ii:
                    if ii == 0:
                        print('st {}, nv {}, jj{}'.format(st,nv,jj))
                        print(ce[nv][ii])
                    state_block += wv.todense()
                    countt += 1
        
        print(np.shape(state_block))
        amean = state_block / countt
            
        plt.imshow(amean)
        plt.suptitle('{}, nV: {}.'.format(st,Nv))
        plt.savefig(loc + name)

        if display:
            plt.show()
        plt.close()

def plotRipsMetric(saveloc = curdir, display=False, nVol = -1, localdir='/aggFiles/', normalize=False):

    volloc = saveloc + localdir
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(volloc)

    ADF = []
    ag = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            
            curtab = tabloc + str(vol) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF.append(pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times']))
        
            afile = volloc + str(vol) + '.pkl'
            with open(afile,'rb') as file:
                ag.append(pickle.load(file))

    Nv = nV = len(ag)
    maxdim = len(ag[0])-1

    print('ag is shape {}.'.format(squareform(ag[0][0]).shape))

    trimLen = int(350)
    goodStates = ['mental','rnd','2bk_body','2bk_faces','2bk_places','2bk_tools']
    goodRepetitions = ['tfMRI_WM_LR','tfMRI_WM_RL','tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL']

    nG = len(goodStates)
    adf = []    
    cr = []
    ce = []
    ct = []
    
    for ni in range(Nv):
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values             
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        aset = [adf[ni].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        ce.append(np.argmax(aset, axis=0))
        ct.append(np.max(aset, axis=0))                          

        for gi, gs in enumerate(goodRepetitions):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values
        repsVec = ADF[ni].index[gv].values
        rdf = ADF[ni].loc[goodVec,:]
        aset = [rdf.loc[:,gs].values[:,np.newaxis] for gs in goodRepetitions]
        cr.append(np.argmax([aset[0]+aset[2], aset[1] + aset[3]], axis=0))
                 

    print(len(ct))
    print(ct[0].shape)

    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   
    
    fig = plt.figure(1,figsize=[18,10])    

    for dim in range(maxdim+1):
        #combos = [np.zeros(Nv)]*int(nG*(nG-1)/2)
        combos = np.zeros((Nv,int(nG*(nG-1)/2 + nG)))
        labels = []
        for ii, (aa,bb) in enumerate(itertools.combinations_with_replacement(np.arange(nG),2)):
            labels.append('d(' + goodStates[aa] + ', ' + goodStates[bb] + ')')
            for ni in range(Nv):
                agi = squareform(ag[ni][dim])
                ca = np.argwhere(np.add(ce[ni]==aa, cr[ni]==0).ravel()).ravel()
                cb = np.argwhere(np.add(ce[ni]==bb, cr[ni]==1).ravel()).ravel()
                combos[ni,ii] = 0
                print('ind ={}, ii = {}, agi.shape = {}'.format(volunteers[ni], ii, agi.shape))
                combos[ni,ii] = np.mean(np.array(agi[ca,:][:,cb]))
        if normalize:
            for ni in range(Nv):
                combos[ni,:] /= np.mean(combos[ni,:])
        
        plt.subplot(maxdim+1,1,dim+1)
        ax = plt.gca()
        ax.axes.set_ylabel('dim {}'.format(dim))
        if dim == maxdim:
            plt.boxplot(combos, labels=labels)
            ax.axes.set_xticklabels(labels,rotation=30)
        else:
            plt.boxplot(combos)

        plt.suptitle('Between-state topological distance, nV: {}.'.format(Nv))
    plt.savefig(loc + 'ConcatenatedIndividualBoxplots.png')

    if display:
        plt.show()
    plt.close()
            
def plotOneEmbedding(saveloc = curdir, display=False, nVol = -1):

    volloc = saveloc + 'UMAP0/'
    tabloc = saveloc + 'train0table/'
    indloc = saveloc + 'ind0/'

    volunteers = getSecondList(volloc)

    ADF = []
    ha = []
    IND = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            curtab = tabloc + str(vol) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF.append(pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times']))
        
            afile = volloc + str(vol) + '.pkl'
            with open(afile,'rb') as file:
                ha.append(pickle.load(file))

            Ifile = indloc + str(vol) + '.npy'
            IND.append(np.load(Ifile,allow_pickle=True))

    Nv = nV = len(ha)
    maxdim = len(ha[0])-1

    trimLen = int(350)
    goodStates = ['mental','rnd','2bk_body','2bk_faces','2bk_places','2bk_tools']
    adf = []    
    ce = []
    ct = []
    
    for ni in range(Nv):
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values             
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        aset = [adf[ni].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        ce.append(np.argmax(aset, axis=0))
        ct.append(np.max(aset, axis=0))                          

    print(len(ct))
    print(ct[0].shape)

    loc = './figures/'
    makedirs(loc ,exist_ok=True)   
    
    nv = 0 
    fig = plt.figure(1,figsize=[18,18])
    dim = -1
    for ax in range(1,(maxdim+1)*3+1,3):
        dim+=1
        plt.subplot(maxdim+1,3,ax)
        X , Y, c, s = [], [], [], []
        X.append(ha[nv][dim][:,0])
        Y.append(ha[nv][dim][:,1])
        c.append(ct[nv].ravel()/np.max(ct[nv].ravel()))
        s.append((ct[nv]!=-11).astype('int')*10)
        plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(), marker='o',alpha=0.5)

        if dim == maxdim:
            cbar = plt.colorbar(orientation='horizontal')
            cbar.set_label('Normalized Experiment Timing',fontsize=16)        

        plt.subplot(maxdim+1,3,ax+1)    
        X , Y, c, s = [], [], [], []
        X.append(ha[nv][dim][:,0])
        Y.append(ha[nv][dim][:,1])
        c.append(ce[nv].ravel())
        s.append((ce[nv]!=-11).astype('int')*10)
        plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5, cmap = 'Set1')
        if dim == maxdim:
            cbar = plt.colorbar(orientation='horizontal', ticks=[0,1,2,3,4,5])
            cbar.set_label('Experiment Type',fontsize=16)
            cbar.ax.set_xticklabels(['mental','rnd','2bk_body','2bk_faces','2bk_places','2bk_tools'],fontsize=10)        

        plt.subplot(maxdim+1,3,ax+2)        
        X , Y, c, s = [], [], [], []
        X.append(ha[nv][dim][:,0])
        Y.append(ha[nv][dim][:,1])
        c.append(ha[nv][dim][:,0]*0+nv+1)
        s.append((ct[nv]!=-11).astype('int')*2)

        doIND = True #False
        if doIND:
            for i in IND[nv][dim]:
                s[-1][i] *=50 
        ax = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5,cmap='Accent')
        #ax.axes.set_xlim(left=-2,right=2)
        #ax.axes.set_ylim(top=2,bottom=-2)
        if dim == maxdim:
            cbar = plt.colorbar(orientation='horizontal')
            cbar.set_label('Volunteer Number',fontsize=16)


    name = (loc + 'OneTestUMAP.png')            
    plt.savefig(name)

    if display:
        plt.show()
    plt.close()

def plotTrainingMetric(saveloc = curdir, display=False, nVols = -1, localdir='grp0/', normalize=False):
       
    buffLen = int(450)
    mixLen = int(300)
    trimLen = int(350)           
    
    goodStates = ['mental','rnd','2bk_body','2bk_faces','2bk_places','2bk_tools']
    goodRepetitions = ['tfMRI_WM_LR','tfMRI_WM_RL','tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL']

    embloc = saveloc + 'UMAP1/'
    hNames = getSecondList(embloc)
    maxdim = len(hNames)-1

    tabloc = saveloc + 'train0table/'
    volloc = saveloc + 'ind0/'
    grploc = saveloc + localdir

    gvolunteers = getSecondList(volloc)
    print('number of possible training group volunteers is {}'.format(len(gvolunteers)))

    if nVols == -1:
        nVols = len(gvolunteers)
    else:
        nVols = nVols
    gvolunteers = gvolunteers[:nVols]

    gADFi = []
    for dim in range(maxdim+1):
        gADFi.append([])

    gADF = {}
    gIt = {}
    for voln in gvolunteers:
        Ifile = saveloc + 'ind0/' + str(voln) + '.npy'

        curtab = tabloc + str(voln) + '.npz'
        tb = np.load(curtab, allow_pickle=True)
        gADF[voln] = pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times'])
        
        gIt[voln] = np.load(Ifile,allow_pickle=True)
        print(voln)
        print(np.histogram(gIt[voln]))
                
    gNv = len(gIt)
    print('n vol in is {}, maxdim is {}'.format(gNv,maxdim))

    gadf = []
    for ni, voln in enumerate(gvolunteers):
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (gADF[voln].loc[:,gs] > -1 ).values
            else:
                gv += (gADF[voln].loc[:,gs] > -1 ).values
        goodVec = sorted(gADF[voln].index[gv].values)
        gadf.append(gADF[voln].loc[goodVec,:])
    
    gce = {}
    gct = {}
    for dim in range(maxdim+1):
        gce[dim] = []
        gct[dim] = []
        for ni, voln in enumerate(gvolunteers):        
            aset = [gadf[ni].loc[:,gs].values[gIt[voln]][:,np.newaxis] for gs in goodStates]
            aset = np.concatenate(aset,axis=1)

            gce[dim].append(np.argmax(aset, axis=1))
            gct[dim].append(np.max(aset, axis=1))
        gce[dim] = np.concatenate(gce[dim], axis=0)
        gct[dim] = np.concatenate(gct[dim], axis=0)

    print(len(gct))
    print(gct[0].shape)
    
    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   
    
    fig = plt.figure(1,figsize=[18,10])    
    #name = (loc + 'ripsMetric_' + st + '.jpg')            

    for dim in range(truncDim,maxdim+1,1):
        gg = np.load(grploc + 'HN.npy')
        combos = [] 
        labels = []
        for ii, (aa,bb) in enumerate(itertools.combinations_with_replacement(np.arange(len(goodStates)),2)):
            labels.append('d(' + goodStates[aa] + ', ' + goodStates[bb] + ')')
            ca = np.argwhere( (gce[dim]==aa).ravel() ).ravel()
            cb = np.argwhere( (gce[dim]==bb).ravel() ).ravel()
            print(' ii = {}, gg.shape = {}'.format( ii, gg.shape))
            if aa==bb:
                combos.append(np.array(gg[ca,:][:,cb][(-np.eye(len(ca))+1).astype(bool)]).ravel())
            else:
                combos.append(np.array(gg[ca,:][:,cb]).ravel())
        
        plt.subplot(maxdim+1,1,dim+1)
        ax = plt.gca()
        ax.axes.set_ylabel('dim {}'.format(dim))
        if dim == maxdim:
            plt.boxplot(combos, labels=labels)
            ax.axes.set_xticklabels(labels,rotation=30)
        else:
            plt.boxplot(combos)

        plt.suptitle('(Training Set) Between-state topological distance, nV: {}.'.format(gNv))
    plt.savefig(loc + 'TrainingSetBoxplot.png')

    if display:
        plt.show()        
    plt.close()


def plotEmbeddingTraining(saveloc = curdir, display=False, nVols = -1, localdir = 'UMAPxyTrainSimplex/', figname='TrainingUMAP.png'):

    buffLen = p_utils.buffLen
    mixLen = p_utils.mixLen
    trimLen = p_utils.trimLen           
    
    goodStates = p_utils.goodStates
    goodRepetitions = p_utils.goodRepetitions

    maxdim = p_utils.maxdim

    tabloc = saveloc + 'train0table/'
    volloc = saveloc + 'ind0/'
    riploc = saveloc + 'rippedTraining/'
    setloc = saveloc + 'setTraining/'

    gvolunteers = getSecondList(setloc)
    print('number of possible training group volunteers is {}'.format(len(gvolunteers)))

    if nVols == -1:
        nVols = len(gvolunteers)
    else:
        nVols = nVols
    gvolunteers = gvolunteers[:nVols]

    gADFi = []
    for dim in range(maxdim+1):
        gADFi.append([])

    gADF = {}
    gIt = {}
    for voln in gvolunteers:
        Ifile = saveloc + 'ind0/' + str(voln) + '.npy'

        curtab = tabloc + str(voln) + '.npz'
        tb = np.load(curtab, allow_pickle=True)
        gADF[voln] = pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times'])
        
        gIt[voln] = np.load(Ifile,allow_pickle=True)
        print(voln)
        print(np.histogram(gIt[voln]))
                
    gNv = len(gIt)
    print('n vol in is {}, maxdim is {}'.format(gNv,maxdim))

    gadf = []
    for ni, voln in enumerate(gvolunteers):
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (gADF[voln].loc[:,gs] > -1 ).values
            else:
                gv += (gADF[voln].loc[:,gs] > -1 ).values
        goodVec = sorted(gADF[voln].index[gv].values)
        gadf.append(gADF[voln].loc[goodVec,:])
    
    gce = {}
    gct = {}
    gcv = {}
    for dim in range(maxdim+1):
        gce[dim] = []
        gct[dim] = []
        gcv[dim] = []
        for ni, voln in enumerate(gvolunteers):        
            aset = [gadf[ni].loc[:,gs].values[gIt[voln]][:,np.newaxis] for gs in goodStates]
            aset = np.concatenate(aset,axis=1)

            gce[dim].append(np.argmax(aset, axis=1))
            gct[dim].append(np.divide( np.max(aset, axis=1), np.array( [ np.max(gadf[ni].loc[:,goodStates[gcei]]) for gcei in gce[dim][-1] ] ) ))
            gcv[dim].append(np.zeros(gct[dim][-1].shape) + ni)
        gce[dim] = np.concatenate(gce[dim], axis=0)
        gct[dim] = np.concatenate(gct[dim], axis=0)
        gcv[dim] = np.concatenate(gcv[dim], axis=0)

    print(len(gct))
    print(gct[0].shape)
    
    embloc = saveloc + localdir
    hNames = getSecondList(embloc)
    maxdim = len(hNames)-1

    h3 = []
    for hi, hn in enumerate(hNames):
        embfile = embloc + str(hn) + '.npy'
        h3.append(np.load(embfile))
        print('h3[{}] is len({}) and type {}'.format(hi,len(h3[hi]),type(h3[hi])))
    
    loc = './figures/'
    makedirs(loc ,exist_ok=True)   

    fig = plt.figure(1,figsize=[40,25])
    for dim in range(maxdim+1):
        ax = 3*dim + 1
        plt.subplot(3,3,ax)
        X , Y, c, s = [], [], [], []
        X.append(h3[dim][:,0])
        Y.append(h3[dim][:,1])
        c.append(gct[dim].ravel())
        s.append((gct[dim]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(), marker='o',alpha=0.5)
        if dim == maxdim:
            cbar = fig.colorbar(ppl,ticks=np.arange(0,1,0.1), orientation='horizontal')
            
        plt.subplot(3,3,ax+1)    
        X , Y, c, s = [], [], [], []
        X.append(h3[dim][:,0])
        Y.append(h3[dim][:,1])
        c.append(gce[dim].ravel())
        s.append((gce[dim]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5, cmap = 'nipy_spectral')
        if dim == maxdim:
            tks = np.arange(len(goodStates))
            cbar = fig.colorbar(ppl, ticks=tks, orientation='horizontal')
            cbar.ax.set_xticklabels(goodStates,rotation=90)
            
        plt.subplot(3,3,ax+2)        
        X , Y, c, s = [], [], [], []
        X.append(h3[dim][:,0])
        Y.append(h3[dim][:,1])
        c.append(gcv[dim].ravel())
        s.append((gct[dim]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5,cmap='gist_ncar')
        if dim == maxdim:
            tks = np.unique(np.array(c).ravel())
            cbar = fig.colorbar(ppl, ticks=tks, orientation='horizontal')
            cbar.ax.set_xticklabels(tks, rotation = 90)        

    name = (loc + figname)            
    plt.savefig(name)

    if display:
        plt.show()
    plt.close()        
     
        
def plotAggUMAPs(saveloc = curdir, figname='SimplexTestingUMAP.png', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/'):

    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1            
            ttime[i,0] = initt
        else:            
            ttime[i,0] = 0
            inite = t
            initt = 0
                                  
    embloc = saveloc + localdir
    tabloc = saveloc + 'test0table/'

    volunteers = getSecondList(embloc)
    
    vloc = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/anat/'
    tvol = []
    for vol in volunteers:

        adata = np.load(vloc + vol ).T

        nvox, nT = nP, nT = np.shape(adata)
        print(nP,nT)
        print('Data histogram')
        print(np.histogram(adata.ravel()))

        testData = np.sum(adata,axis=1)

        if sum(testData==0)==0:
            tvol.append(vol)

    h3 = []
    ADF = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
        
            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))
            ADF.append(pd.DataFrame(ttime[trimLen:-trimLen],columns=[0]))

    Nv = nV = len(h3)
    nT = h3[vi][0].shape[0]

    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    adf = []    
    ce = []
    ct = []
    cv = []
    
    for ni in range(Nv):
        gv = ADF[ni].loc[:,0] > -1              
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])                          
        cv.append(np.zeros(ct[-1].shape)+ni)
        print(np.histogram(ce[-1]))
        print(np.histogram(ct[-1]))

    print(len(ct))
    print(ct[0].shape)

    fig = plt.figure(1,figsize=[40,25])
    dim = -1#p_utils.truncDim-1
    for ax in range(1,10,3):
        dim+=1
        if dim>maxdim:
            break
        plt.subplot(3,3,ax)
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(ct[nv].ravel())
            s.append((ct[nv]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(), marker='o',alpha=0.5, cmap='viridis')
        if dim == maxdim:
            cbar = fig.colorbar(ppl,ticks=np.arange(0,1,0.1), orientation='horizontal')
            
        plt.subplot(3,3,ax+1)    
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(ce[nv].ravel())
            s.append((ce[nv]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5, cmap = 'nipy_spectral')
        if dim == maxdim:
            tks = np.arange(len(goodStates))
            cbar = fig.colorbar(ppl, ticks=tks, orientation='horizontal')
            cbar.ax.set_xticklabels(goodStates,rotation=90)
            
        plt.subplot(3,3,ax+2)        
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(cv[nv])
            s.append((ct[nv]!=-11).astype('int')*20)
        ppl = plt.scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='o',alpha=0.5,cmap='viridis')
        if dim == maxdim:
            tks = np.arange(Nv)
            cbar = fig.colorbar(ppl, ticks=tks, orientation='horizontal')
            cbar.ax.set_xticklabels(tks, rotation = 90)        

    loc = './figures/'
    makedirs(loc ,exist_ok=True)   
    name = (loc + figname)            
    plt.savefig(name)

    if display:
        plt.show()
    plt.close()

def plotAggMovies(saveloc = curdir, display=False, nVol = -1, localdir = 'Testing_Embeds/'):
    import matplotlib
    matplotlib.use('Agg')

    embloc = saveloc + localdir
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(embloc)

    ADF = []
    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            curtab = tabloc + str(vol) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF.append(pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times']))
        
            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}'.format(vi,len(h3[vi]),type(h3[vi])))            

    h4 = np.concatenate(h3,axis=0)
    print('h4 is len {} and type {}.'.format(len(h4), type(h4)))
    print('h4 is shape {}.'.format(h4.shape))

    Nv = nV = len(h3)
    
    adf = []    
    ce = []
    ct = []
    cv = []
    
    for ni in range(Nv):
        
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values             
                
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        aset = [adf[ni].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        ce.extend(np.argmax(aset, axis=0))
        ct.extend(np.max(aset, axis=0))                          
        cv.extend(np.zeros([adf[ni].shape[0],1])+ni)

    ce = np.array(ce)
    ct = np.array(ct)
    cv = np.array(cv)
    print('regarding ct')
    print(ct.shape)   
    print('regarding cv')
    print(cv.shape)

    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[40,25])
    dim = 0
    
    axes[0, 0].set_title('Density')
    axes[0, 0].hist2d(h4[:, 0], h4[:, 1], bins=100)

    X , Y, c, s = [], [], [], []
    X.append(h4[:,0])
    Y.append(h4[:,1])
    c.append(ct.ravel())
    s.append((ct!=-11).astype('int')*20)
    axes[0, 1].set_title('Timing')
    axes[0, 1].scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(), marker='o',alpha=0.5)

    X , Y, c, s = [], [], [], []    
    X.append(h4[:,0])
    Y.append(h4[:,1])
    c.append(ce.ravel())
    s.append((ce!=-11).astype('int')*20)
    axes[1, 0].set_title('Conditions')        
    axes[1, 0].scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                marker='o',alpha=0.5, cmap = 'tab20c')

    X , Y, c, s = [], [], [], []
    X.append(h4[:,0])
    Y.append(h4[:,1])
    c.append(cv)
    s.append((ct!=-11).astype('int')*20)
    axes[1, 1].set_title('Volunteers')        
    axes[1, 1].scatter(np.concatenate(X).ravel(),np.concatenate(Y).ravel(),
                c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                marker='o',alpha=0.5,cmap='Set1')

    axes = axes.flat
    dots = []
    for ax in axes:
        vec = (ce==0) & (ct==0)
        vec = vec.ravel()     
        colors = cv[vec].ravel()     
        print(vec.shape)
        print(h4.shape)
        data = h4[vec,:]
        print(data.shape)
        img = ax.scatter(data[:,0],data[:,1],c=colors,marker='o',cmap='Set1')
        dots.append(img)
        
    fig.tight_layout()    

    plt.show()

    for gi, gs in enumerate(goodStates):
            
        metadata = dict(title='TimeseriesAnimation_{}'.format(gs), artist='Matplotlib')
        print(gs)
        writer = FFMpegWriter(fps=3, metadata=metadata)
        
        with writer.saving(fig, loc + 'TimeseriesAnimation_{}.mp4'.format(gs), 100):
            for tt in range(int(np.max(ct[ce==gi]))):
                for dot in dots:
                    vec = (ce==gi) & (ct==tt)
                    vec = vec.ravel()
                    dot.set_offsets(h4[vec,:])
                writer.grab_frame()

    plt.close(fig)
    
def plotGroupMetric(saveloc = curdir, display=False, nVol = -1, localdir='Testing_Distances/', normalize=False):
       
    buffLen = int(450)
    mixLen = int(300)
    trimLen = int(350)           
    
    goodStates = p_utils.goodStates
    goodRepetitions = p_utils.goodRepetitions

    embloc = saveloc + 'UMAP1/'
    hNames = p_utils.getSecondList(embloc)
    maxdim = p_utils.maxdim-2
    truncDim = p_utils.truncDim

    tabloc = saveloc + 'train0table/'
    volloc = saveloc + 'ind0/'
    gvolunteers = p_utils.getSecondList(volloc)
    print('number of possible training group volunteers is {}'.format(len(gvolunteers)))

    gADFi = []
    for dim in range(maxdim+1):
        gADFi.append([])

    gADF = {}
    gIt = {}
    for voln in gvolunteers:
        Ifile = saveloc + 'ind0/' + str(voln) + '.npy'

        curtab = tabloc + str(voln) + '.npz'
        tb = np.load(curtab, allow_pickle=True)
        gADF[voln] = pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times'])
        
        gIt[voln] = [np.load(Ifile,allow_pickle=True)]
        print(voln)

    gNv = len(gIt)
    print('n vol in is {}, maxdim is {}'.format(gNv,maxdim))   
    
    gadf = {} 
    for ni, voln in enumerate(gvolunteers):
        gadf[voln] = gADF[voln]
        '''
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (gADF[voln].loc[:,gs] > -1 ).values
            else:
                gv += (gADF[voln].loc[:,gs] > -1 ).values
        goodVec = sorted(gADF[voln].index[gv].values)
        gadf[voln] = gADF[voln].loc[goodVec,:]
        '''

    gce = {}
    gct = {}
    
    for dim in range(truncDim,maxdim+1):
        gce[dim] = []
        gct[dim] = []
        for ni, voln in enumerate(gvolunteers):
            aset = [gadf[voln].loc[:,gs].values[gIt[voln][dim]][:,np.newaxis] for gs in goodStates]
            print('shape aset[0] & aset[1] init is {} & {}'.format(aset[0].shape, aset[1].shape))
            aset = np.concatenate(aset,axis=1)
            print('aset final is shape {}.'.format(aset.shape))
            
            gce[dim].append(np.argmax(aset, axis=1))
            gct[dim].append(np.max(aset, axis=1))
        gce[dim] = np.concatenate(gce[dim], axis=0)
        gct[dim] = np.concatenate(gct[dim], axis=0)

        print('for dim {}, len gce is {}'.format(dim,len(gce[dim])))

    volloc = saveloc + localdir 
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(volloc)
    #volunteers = list(set(volunteers)-set(gvolunteers))

    ADF = {}
    ag = {}
    for vi, voln in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            
            curtab = tabloc + str(voln) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF[voln] = pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times'])
        
            afile = volloc + str(voln) + '.pkl'
            with open(afile,'rb') as file:                
                ag[voln] = {}                
                temp = [pickle.load(file)]
                for i, dim in enumerate(range(truncDim,maxdim+1,1)):
                    ag[voln][dim] = temp[dim]

            lastvol = voln

    Nv = nV = len(ag)
    print('num volunteers is {}'.format(Nv))
    print(list(ag.keys()))
    print(lastvol)
    print(voln)
    print(len(ag[lastvol]))
    print(type(ag[lastvol]))

    print('ag[lastvol][0] is shape {}.'.format(ag[lastvol][0].shape))

    nG = len(goodStates)
    adf = {}    
    ce = {}
    ct = {}
    
    for ni, voln in enumerate(volunteers):
        adf[voln] = ADF[voln]
        '''
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[voln].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[voln].loc[:,gs] > -1 ).values             
                #gv = gv or (ADF[voln].loc[:,gs] > -1 ).values  
        goodVec = ADF[voln].index[gv].values
        adf[voln] = ADF[voln].loc[goodVec,:]
        '''
        
        aset = [adf[voln].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        aset = np.concatenate(aset,axis=1)
        ce[voln] = np.argmax(aset, axis=1)
        ct[voln] = np.max(aset, axis=1)                        

    print(np.histogram(ce[voln]))
    print(len(ct))
    print(ct[voln].shape)

    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   
    
    fig = plt.figure(1,figsize=[18,10])    

    for dim in range(truncDim,maxdim+1):
        #combos = [np.zeros(Nv)]*int(nG*(nG-1)/2)
        combos = np.zeros((Nv,int(nG*(nG-1)/2 + nG)))
        labels = []
        for ii, (aa,bb) in enumerate(itertools.combinations_with_replacement(np.arange(nG),2)):
            labels.append('d(v.' + goodStates[aa] + ', g.' + goodStates[bb] + ')')
            for ni, voln in enumerate(volunteers):
                agi = ag[voln][dim]
                ca = np.argwhere( (ce[voln]==aa).ravel() ).ravel()
                cb = np.argwhere( (gce[dim]==bb).ravel() ).ravel()
                #combos[ni,ii] = 0
                #print('ind ={}, ii = {}, agi.shape = {}'.format(volunteers[ni], ii, agi.shape))
                combos[ni,ii] = np.mean(agi[ca,:][:,cb].ravel())
        if normalize:
            for ni in range(Nv):
                combos[ni,:] /= np.mean(combos[ni,:])
        
        plt.subplot(maxdim+1,1,dim+1)
        ax = plt.gca()
        ax.axes.set_ylabel('dim {}'.format(dim))
        if dim == maxdim:
            plt.boxplot(combos, labels=labels)
            ax.axes.set_xticklabels(labels,rotation=30)
        else:
            plt.boxplot(combos)

        plt.suptitle('Between-state topological distance, nV: {}.'.format(Nv))
    plt.savefig(loc + 'GroupNativeDimBoxplot.png')

    if display:
        plt.show()
    plt.close()

def plotEmbeddingBoxplot(saveloc = curdir, display=False, nVol = -1):

    embloc = saveloc + 'Testing_Embeds/'
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(embloc)

    ADF = {}
    h3 = {}
    for vi, voln in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            curtab = tabloc + str(voln) + '.npz'
            tb = np.load(curtab, allow_pickle=True)
            ADF[voln] = pd.DataFrame(tb['table'],columns=tb['states'],index=tb['times'])
        
            embfile = embloc + str(voln) + '.pkl'
            with open(embfile,'rb') as file:
                h3[voln] = [pickle.load(file)]
            print('h3[{}] is len {}, and type {}'.format(voln, len(h3[voln]), type(h3[voln])))
            print('h3[voln][0] is shape {}'.format(h3[voln][0].shape))

    Nv = nV = len(h3)

    trimLen = int(350)
    goodStates = p_utils.goodStates
    nG = len(goodStates)
    adf = {}
    ce = {}
    ct = {}
    
    for ni, voln in enumerate(volunteers):
        adf[voln] = ADF[voln]
        
        '''
        for gi, gs in enumerate(goodStates):
            if gi == 0:
                gv = (ADF[ni].loc[:,gs] > -1 ).values
            else:
                gv += (ADF[ni].loc[:,gs] > -1 ).values             
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        '''
        
        aset = [adf[voln].loc[:,gs].values[:,np.newaxis] for gs in goodStates]
        
        print('shape aset[0] & aset[1] init is {} & {}'.format(aset[0].shape, aset[1].shape))
        aset = np.concatenate(aset,axis=1)
        print('aset final is shape {}.'.format(aset.shape))
        
        ce[voln] = np.argmax(aset, axis=1)
        ct[voln] = np.max(aset, axis=1)
        
        print('ce[voln] is shape {}. It''s histogram is : '.format(ce[voln].shape))
        print(np.histogram(ce[voln]))
    
    print('len ct is {}'.format(len(ct)))
    print('ct shape is {}'.format(ct[voln].shape))

    loc = saveloc + '/figures/'
    makedirs(loc ,exist_ok=True)   
    
    fig = plt.figure(1,figsize=[18,10])    

    dim=0
    if dim==0:#in range(truncDim, maxdim+1):
        print('for dimension {}'.format(dim))
        combos = []
        labels = []
        for ii, (aa,bb) in enumerate(itertools.combinations_with_replacement(np.arange(nG),2)):
            labels.append('emb_d(' + goodStates[aa] + ', ' + goodStates[bb] + ')')
            
            tempA = []
            tempB = []
            for nv, voln in enumerate(volunteers):
                ca = np.argwhere( (ce[voln]==aa).ravel() ).ravel()
                cb = np.argwhere( (ce[voln]==bb).ravel() ).ravel()
                
                tempA.append(h3[voln][dim][ca,:])                
                tempB.append(h3[voln][dim][cb,:])
            tempA = np.concatenate(tempA,axis=0)
            tempB = np.concatenate(tempB,axis=0)

            if aa==bb:
                combos.append(pdist(tempA,metric='euclidean').ravel())
            else:
                combos.append(cdist(tempA,tempB,metric='euclidean').ravel())
                
        plt.subplot(maxdim+1,1,dim+1)
        ax = plt.gca()
        ax.axes.set_ylabel('dim {}'.format(dim))
        if dim == maxdim:
            plt.boxplot(combos, labels=labels)
            ax.axes.set_xticklabels(labels,rotation=30)
        else:
            plt.boxplot(combos)

        plt.suptitle('Between-state embedding distance, nV: {}.'.format(Nv))

    name = (loc + 'GroupUMAPDimBoxplot.png')            
    plt.savefig(name)

    if display:
        plt.show()

    plt.close()

def plotSSIM(saveloc = curdir, savename='SimplexTestingSSIM', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False): 

    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1            
            ttime[i,0] = initt
        else:            
            ttime[i,0] = 0
            inite = t
            initt = 0
    print('ttime hist')
    print(np.histogram(ttime))
                                  
    embloc = saveloc + localdir
    tabloc = saveloc + 'test0table/'

    volunteers = getSecondList(embloc)

    h3 = []
    ADF = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
        
            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))
            ADF.append(pd.DataFrame(ttime[trimLen:-trimLen],columns=[0]))

    Nv = len(volunteers)    
    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    adf = []    
    ce = []
    ct = []
    cv = []
    
    for ni in range(Nv):
        gv = ADF[ni].loc[:,0] > -1              
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])
        
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])                          
        cv.append(np.zeros(ct[-1].shape)+ni)
        print(np.histogram(ce[-1]))
        print(np.histogram(ct[-1]))


    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)

    def setFed(data):
        x = data[:,0]
        y = data[:,1]
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)# Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:128j, ymin:ymax:128j]

        return xx, yy
 

    def getFed(data,xx=None,yy=None):
        x = data[:,0]
        y = data[:,1]

        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values,bw_method=0.03)
        kernel.factor

        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        return f

    types_ce = np.unique(np.concatenate(ce).ravel())
    print('typrs_ve = {} '.format(types_ce))

    Floc = saveloc + 'figures/' + savename + 'In/'
    makedirs(Floc ,exist_ok=True)

    Sloc = saveloc + 'figures/' + savename + 'Out/'
    makedirs(Sloc ,exist_ok=True)

    if precalc:        
        
        for dim in range(maxdim+1):
            Ffile = (Floc + 'H' + str(dim) + '.pkl')
            with open(Ffile, 'rb') as file:
                F = pickle.load(file)
            
            vv=[]
            ee=[]
            H = []
            counter = -1    

            xx,yy = setFed(np.concatenate([h3[nv][dim] for nv in range(Nv)]))

            print('Dim = {}'.format(dim))
            data = []
            for nv in range(Nv):
                name = 'Vol_{}'.format(nv)
                print([dim, name])
                data.append(h3[nv][dim])
                #F[name] = getFed(data[-1], xx=xx, yy=yy)
                counter += 1
                vv.append(counter)
                H.append(name)

            Data = np.concatenate(data)    
            print('For dim {}, Data is shape {}'.format(dim,Data.shape))

            for te in types_ce:
                name = '{}_{}'.format(te,goodStates[te])
                svec = np.concatenate([ce[nv]==te for nv in range(Nv)])
                print([dim, name, 'sum points =', sum(svec)])
                print('svec is shape {}, min is, {}, max {}'.format(svec.shape,np.min(svec),np.max(svec)))
                #F[name] = getFed(Data[svec.ravel(),:], xx=xx, yy=yy)
                one_ce = ce[0]==te
                types_ct = np.unique(ct[0][one_ce.ravel()])
                counter += 1
                ee.append(counter)
                H.append(name)

                for ty in types_ct:
                    name = '{}_{}_t={}'.format(te,goodStates[te],int(ty))
                    tvec = np.concatenate([ct[nv]==ty for nv in range(Nv)])
                    cvec = (np.multiply(svec, tvec)).ravel()
                    print([dim, name, 'sum points =', sum(cvec)])
                    #F[name] = getFed(Data[cvec,:], xx=xx, yy=yy)
                    counter += 1
                    H.append(name)

            pool = Pool(14)
            ssim_fun = partial(ssim, full=False, gradient=False)
            S = list(pool.starmap(ssim_fun, tqdm(list([F[aa]/np.max(F[aa]), F[bb]/np.max(F[bb])] for aa,bb in itertools.combinations(F,2)), total=int(len(F)*(len(F)-1)/2) )))
            pool.close()

            Sfile = (Sloc + 'H' + str(dim) + '.pkl')
            with open(Sfile, 'wb') as file:
                pickle.dump(S,file, pickle.HIGHEST_PROTOCOL)  

            fig=plt.figure(figsize=[12,12])
            ax=plt.subplot(111)
            img=plt.imshow(squareform(S))
            plt.xticks([0] + ee, ['Vols'] + [H[eee] for eee in ee],rotation='vertical')
            plt.colorbar()

            name = saveloc + 'figures/' + savename + '_H' + str(dim) + '.png'
            plt.savefig(name)

            if display:
                plt.show()

            plt.close()            

    else:
        for dim in range(maxdim+1):
            F = {}
            vv=[]
            ee=[]
            H = []
            counter = -1    

            xx,yy = setFed(np.concatenate([h3[nv][dim] for nv in range(Nv)]))

            print('Dim = {}'.format(dim))
            data = []
            for nv in range(Nv):
                name = 'Vol_{}'.format(nv)
                print([dim, name])
                data.append(h3[nv][dim])
                F[name] = getFed(data[-1], xx=xx, yy=yy)
                counter += 1
                vv.append(counter)
                H.append(name)

            Data = np.concatenate(data)    
            print('For dim {}, Data is shape {}'.format(dim,Data.shape))

            for te in types_ce:
                name = '{}_{}'.format(te,goodStates[te])
                svec = np.concatenate([ce[nv]==te for nv in range(Nv)]).ravel()
                print([dim, name, 'sum points =', sum(svec)])
                print('svec is shape {}, min is, {}, max {}'.format(svec.shape,np.min(svec),np.max(svec)))
                F[name] = getFed(Data[svec.ravel(),:], xx=xx, yy=yy)
                one_ce = ce[0]==te
                types_ct = np.unique(ct[0][one_ce.ravel()])
                print('types_ct hist')
                print(np.histogram(types_ct))
                counter += 1
                ee.append(counter)
                H.append(name)

                for ty in types_ct:
                    name = '{}_{}_t={}'.format(te,goodStates[te],int(ty))
                    tvec = np.concatenate([ct[nv]==ty for nv in range(Nv)]).ravel()
                    cvec = (np.multiply(svec, tvec)).ravel()
                    print([dim, name, 'sum points =', sum(cvec)])
                    F[name] = getFed(Data[cvec,:], xx=xx, yy=yy)
                    counter += 1
                    H.append(name)

            Ffile = (Floc + 'H' + str(dim) + '.pkl')
            with open(Ffile, 'wb') as file:
                pickle.dump(F,file, pickle.HIGHEST_PROTOCOL)  
                    
            pool = Pool(14)
            ssim_fun = partial(ssim, full=False, gradient=False)
            S = list(pool.starmap(ssim_fun, tqdm(list([F[aa]/np.max(F[aa]), F[bb]/np.max(F[bb])] for aa,bb in itertools.combinations(F,2)), total=int(len(F)*(len(F)-1)/2) )))
            pool.close()

            Sfile = (Sloc + 'H' + str(dim) + '.pkl')
            with open(Sfile, 'wb') as file:
                pickle.dump(S,file, pickle.HIGHEST_PROTOCOL)  

            fig=plt.figure(figsize=[12,12])
            ax=plt.subplot(111)
            img=plt.imshow(squareform(S))
            plt.xticks([0] + ee, ['Vols'] + [H[eee] for eee in ee],rotation='vertical')
            plt.colorbar()

            name = saveloc + 'figures/' + savename + '_H' + str(dim) + '.png'
            plt.savefig(name)

            if display:
                plt.show()

            plt.close()


            
def plotStateMap(saveloc = curdir, savename='SimplexTestingStates', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False):

    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1
            ttime[i,0] = initt
        else:
            ttime[i,0] = 0
            inite = t
            initt = 0
    print('ttime hist')
    print(np.histogram(ttime))

    trimLen = p_utils.trimLen

    embloc = saveloc + localdir
    tabloc = saveloc + 'test0table/'

    volunteers = getSecondList(embloc)

    ADF = []
    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:

            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))
            ADF.append(pd.DataFrame(ttime[trimLen:-trimLen],columns=[0]))

    Nv = nV = len(h3)

    Nv = len(volunteers)
    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    adf = []
    ce = []
    ct = []
    cv = []

    for ni in range(Nv):
        gv = ADF[ni].loc[:,0] > -1
        goodVec = ADF[ni].index[gv].values
        adf.append(ADF[ni].loc[goodVec,:])

        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])
        cv.append(np.zeros(ct[-1].shape)+ni)
        print(np.histogram(ce[-1]))
        print(np.histogram(ct[-1]))

    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)

    def setFed(data):
        x = data[:,0]
        y = data[:,1]
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)# Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:128j, ymin:ymax:128j]

        return xx, yy, xmin, xmax, ymin, ymax
 

    def getFed(data,xx=None,yy=None):
        x = data[:,0]
        y = data[:,1]

        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values,bw_method=0.03)
        kernel.factor

        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        return f

    CE = np.concatenate(ce).ravel()
    print(['CE shape is: ', CE.shape])
    types_ce = np.unique(CE)
    
    Floc = saveloc + 'figures/' + savename + 'In/'
    makedirs(Floc ,exist_ok=True)

    Sloc = saveloc + 'figures/' + savename + 'Out/'
    makedirs(Sloc ,exist_ok=True)

    ImgOuts = {}

    for dim in range(maxdim+1):
        F = {}
        vv=[]
        ee=[]
        H = []
        counter = -1    

        data = np.concatenate([h3[nv][dim] for nv in range(Nv)])
        xx, yy, xmin, xmax, ymin, ymax = setFed(data)
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(xmin,xmax,num=len(xx)))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(ymin,ymax,num=len(yy)))
        ysnap = list(map(snap_fun,data[:,1]))
        points = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        

        F = getFed(data, xx=xx, yy=yy)
        
        wtr = watershed(1-F)
        uwtr = np.unique(wtr)
        print(uwtr)

        sig = 0.05
        bsig = 0.05/(len(uwtr)*len(types_ce))
        print('Bonferroni sig = ' + str(bsig))
        
        wpolys = {}
        wouts = {}
        routs = pd.DataFrame(columns=types_ce,dtype=float)
        rmeans = pd.DataFrame(columns=types_ce,dtype=float)
        rstds = pd.DataFrame(columns=types_ce,dtype=float)
        scounts = pd.DataFrame(columns=types_ce,dtype=float)
        ImgOuts[dim] = -np.ones(wtr.shape)

        for w in uwtr:
            print('Doing wtr region # ' + str(w))
            rows, cols = np.nonzero(wtr==w)
            pts = [[xx[rows[i],0], yy[0,cols[i]]] for i in range(len(rows)) ]
            obj = geo.MultiPoint(pts)
            wpolys[w] = obj.convex_hull
            wouts[w] = [i for i,k in enumerate(points) if k.within(wpolys[w])]        

            #print(wouts[w])
            #print(len(wouts[w]))

            if not wouts[w]:
                print('no points under wtr region # ' + str(w))
                routs.loc[w,:] = np.nan
                continue

            rands = [np.random.choice(CE, size=len(wouts[w]), replace=False) for _ in range(200)]
            for e in types_ce:
                dist = np.array([np.sum(rr==e) for rr in rands]).ravel()
                obs = np.sum([CE[ind]==e for ind in wouts[w]])
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd
                p_value = st.norm.sf(zstat)
                rmeans.loc[w,e] = rmea
                rstds.loc[w,e] = rstd
                scounts.loc[w,e] = obs
                routs.loc[w,e] = p_value

            if any(routs.loc[w,:].values < bsig):
                e = routs.columns.values[np.argmin(routs.loc[w,:].values)]
                print([e, ':', routs.loc[w,:].values.ravel(), ':', bsig, ':' ,routs.loc[w,e]])
            else:
                e=-1
                print([e, ':', routs.loc[w,:].values.ravel()])
            ImgOuts[dim][wtr==w] = e

        print(np.histogram(ImgOuts[dim]))
        name = './figures/' + savename + '_H' + str(dim) + '.png'
        plt.imshow(ImgOuts[dim], cmap='nipy_spectral')
        cbar = plt.colorbar()
        cbar.set_ticks([-1] + list(types_ce))
        cbar.set_ticklabels(goodStates)
        plt.suptitle( (savename + ' | nVol {} | Bonferroni sig {:.3f}'.format( Nv, bsig)) )
        plt.savefig(name)
        plt.close()
        
def plotCombinedPlots(saveloc = curdir, savename='SimplexTestingStates', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False):        

    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1
            ttime[i,0] = initt
        else:
            ttime[i,0] = 0
            inite = t
            initt = 0
    print('ttime hist')
    print(np.histogram(ttime))
    print(ttime.shape)
    print(ttime)

    trimLen = p_utils.trimLen

    embloc = saveloc + localdir
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(embloc)
    '''
    tabloc0 = saveloc + 'train0table/'
    volunteers0 = getSecondList(tabloc0)
    volunteers = list(set(volunteers0).intersection(set(volunteers1)))
    print(volunteers)
    '''

    ADF = []
    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:

            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))
            ADF.append(pd.DataFrame(ttime[trimLen:-trimLen],columns=[0]))

    Nv = nV = len(h3)

    Nv = len(volunteers)
    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    ce = []
    ct = []
    cv = []

    for ni in range(Nv):
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])
        cv.append(np.zeros(ct[-1].shape)+ni)
        print(np.histogram(ce[-1]))
        print(np.histogram(ct[-1]))

    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)

    def setFed(data):
        x = data[:,0]
        y = data[:,1]
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)# Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:256j, ymin:ymax:256j]
        #xx, yy = np.mgrid[xmin:xmax:56j, ymin:ymax:56j]

        return xx, yy, xmin, xmax, ymin, ymax

    def getFed(data,xx=None,yy=None):
        x = data[:,0]
        y = data[:,1]

        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values,bw_method=0.08)
        kFactor = kernel.factor

        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        return f, kFactor
    
    CE = np.concatenate(ce).ravel()
    print(['CE shape is: ', CE.shape])
    types_ce = np.unique(CE)
    # Set index boundaries for list of events
    mxe1 = np.max(types_ce)+1
    mne = 0
    
    CV = np.concatenate(cv).ravel()
    print(['CV shape is: ', CV.shape])
    types_cv = np.unique(CV)
    # set index boundaries for list of volunteers
    mxv1 = np.max(types_cv)+1
    mnv = -1

    CT = np.concatenate(ct).ravel()

    Floc = saveloc + 'figures/' + savename + 'In/'
    makedirs(Floc ,exist_ok=True)

    Sloc = saveloc + 'figures/' + savename + 'Out/'
    makedirs(Sloc ,exist_ok=True)

    e_ImgOuts = {}
    v_ImgOuts = {}
    e_ImgOuts_best = {}
    v_ImgOuts_best = {}
    
    wSaves = {}
    pSaves = {}
    
    aSaves = {}
    
    #For experiments

    for dim in range(maxdim+1):
        F = {}
        vv = []
        ee = []
        H = []
        counter = -1    

        data = np.concatenate([h3[nv][dim] for nv in range(Nv)])
        print('data is shape {}'.format(data.shape))
        data_vec = np.arange(data.shape[0])
        xx, yy, xmin, xmax, ymin, ymax = setFed(data)
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(xmin,xmax,num=len(xx)))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(ymin,ymax,num=len(yy)))
        ysnap = list(map(snap_fun,data[:,1]))
        points = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        

        F, kFactor = getFed(data, xx=xx, yy=yy)

        wtr = watershed(1-F)
        uwtr = np.unique(wtr)
        #print(uwtr)
        
        aSaves[dim] = {'grid':[xx,yy,xmin,xmax,ymin,ymax], 'F':F, 'wtr':wtr}

        sig = 0.05
        esig = 0.05/(len(uwtr)*len(types_ce))
        vsig = 0.05/(len(uwtr)*len(types_cv))

        e_routs = pd.DataFrame(columns=types_ce,dtype=float)
        e_ImgOuts[dim] = np.zeros(wtr.shape)
        e_ImgOuts_best[dim] = np.zeros(wtr.shape)
        
        v_routs = pd.DataFrame(columns=types_cv,dtype=float)
        v_ImgOuts[dim] = -np.ones(wtr.shape)
        v_ImgOuts_best[dim] = -np.ones(wtr.shape)
        
        wSaves[dim] = {}
        pSaves[dim] = {}
        
        barChart = {}
        barChart_full = {}
        barChart_temp = {}
        boxPlot = np.full((max(uwtr)+1,max(types_ce)+1),np.nan).astype('float')
        boxPlot[:,0] = np.arange(max(uwtr)+1)
        boxPlot_all = np.full((max(uwtr)+1,max(types_ce)+1),0).astype('float')
        boxPlot_all[:,0] = np.arange(max(uwtr)+1)
        print(boxPlot_all.shape)
        proportions = np.full((max(uwtr)+1,max(types_ce)+1),np.nan).astype('float')
        proportions[:,0] = np.arange(max(uwtr)+1)
        proportions_all = np.full((max(uwtr)+1,max(types_ce)+1),0).astype('float')
        proportions_all[:,0] = np.arange(max(uwtr)+1)
        proportions_ref = {}
        lineDraw = {}
        lineDraw_full = {}
        lineDraw_temp = {}
        for e in types_ce:
            barChart[e] = 0            
            barChart_full[e] = 0                        
            lineDraw[e] = set()
            lineDraw_full[e] = set()
            proportions_ref[e] = np.sum(CE==e)

        for w in uwtr:
            #print('Doing wtr region # ' + str(w))
            rows, cols = np.nonzero(wtr==w)
            pts = [[xx[rows[i],0], yy[0,cols[i]]] for i in range(len(rows)) ]
            obj = geo.MultiPoint(pts)
            wpolys = obj.convex_hull
            wouts = [i for i,k in enumerate(points) if k.intersects(wpolys)]        
            wSaves[dim][w] = wouts

            if not wouts:
                print('no points under wtr region # ' + str(w))
                e_routs.loc[w,:] = np.nan 
                v_routs.loc[w,:] = np.nan 
                continue
            else:
                print('Site {} of watershed holds {} points'.format(w,len(wouts)))
                               
            rands = [np.random.choice(CE, size=len(wouts), replace=False) for _ in range(200)]
            for e in types_ce:
                dist = np.array([np.sum(rr==e) for rr in rands]).ravel()
                epts = np.array([CE[pt]==e for pt in wouts])
                wpts = np.array(wouts)[epts]
                obs = np.sum(epts)
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd                
                p_value = st.norm.sf(zstat)
                e_routs.loc[w,e] = p_value            
                barChart_temp[e] = obs
                boxPlot_all[w,e] = obs/rmea
                proportions_all[w,e] = obs/proportions_ref[e]
                lineDraw_temp[e] = np.array(data_vec[wpts])                
                
            rands = [np.random.choice(CV, size=len(wouts), replace=False) for _ in range(200)]
            for v in types_cv:
                dist = np.array([np.sum(rr==v) for rr in rands]).ravel()
                obs = np.sum([CV[ind]==v for ind in wouts])
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd
                p_value = st.norm.sf(zstat)
                v_routs.loc[w,v] = p_value
                
            if any(e_routs.loc[w,:].values < esig):
                
                e_all = e_routs.columns.values[np.array(e_routs.loc[w,:].values < esig).astype('bool')]
                for ei in list(e_all):
                    boxPlot[w,ei] = boxPlot_all[w,ei]
                    barChart_full[ei] += barChart_temp[ei]                    
                    proportions[w,ei] = proportions_all[w,ei]
                    lineDraw_full[ei].update(list(lineDraw_temp[ei]))                                    
                    
                e_best = e_routs.columns.values[np.argmin(e_routs.loc[w,:].values)]
                if np.sum(e_routs.loc[w,:].values < esig)>1:
                    e = mxe1
                else:
                    e = e_best
                    barChart[e] += barChart_temp[e]                    
                    lineDraw[e].update(list(lineDraw_temp[e]))                
            else:
                e_best = e = mne
            e_ImgOuts[dim][wtr==w] = e
            e_ImgOuts_best[dim][wtr==w] = e_best

            if any(v_routs.loc[w,:].values < vsig):
                v_best = v_routs.columns.values[np.argmin(v_routs.loc[w,:].values)]
                vsum = np.sum(v_routs.loc[w,:].values < vsig)
                if vsum>1:
                    v = vsum#mxv1
                else:
                    v = 1#v_best
            else:
                v_best=v=mnv
            v_ImgOuts[dim][wtr==w] = v
            v_ImgOuts_best[dim][wtr==w] = v_best

        print('For experiments, total segments = {}'.format(len(e_ImgOuts[dim].ravel())) )
        print(np.histogram(e_ImgOuts[dim]))
        print('For Volunteers, total segments = {}'.format(len(v_ImgOuts[dim].ravel())) )
        print(np.histogram(v_ImgOuts[dim]))
        
        print('bp all hist')
        print(np.histogram(boxPlot_all[:,1:]))
    
        pSaves[dim]['e'] = e_routs
        pSaves[dim]['v'] = v_routs
    
        makedirs('./figures/' ,exist_ok=True)
        
        name = './figures/' + savename + '_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        
        ax = plt.subplot(221) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(ce[nv].ravel())
            s.append((ce[nv]!=-11).astype('int')*10)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.5,cmap='nipy_spectral',vmin=0,vmax=mxe1)
        #cnt = ax.contour(yy,-xx,F)
        ax.set_aspect('equal','box')     
        
        ax = plt.subplot(222)
        img = ax.imshow(e_ImgOuts[dim], cmap='nipy_spectral',alpha=0.5,vmin=0,vmax=mxe1)
        cbar = plt.colorbar(img, orientation='horizontal')
        cbar.set_ticks([0] + list(types_ce) + [mxe1])
        cbar.ax.set_xticklabels( goodStates + ['Multi'], rotation=90)
        ax.contour(F,alpha=0.5)        
        
        ax = plt.subplot(223) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(cv[nv])
            s.append((ct[nv]!=-11).astype('int')*10)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.5,cmap='viridis',vmin=-1,vmax=mxv1)
        #cnt = ax.contour(yy,-xx,F)
        ax.set_aspect('equal','box')            
        
        ax = plt.subplot(224)
        img = ax.imshow(v_ImgOuts[dim], cmap='viridis',alpha=0.5,vmin=-1,vmax=mxv1)
        cbar = plt.colorbar(img, orientation='horizontal')        
        #cbar.set_ticks( [-1] + list(types_cv) + [mxv1] )
        #cbar.ax.set_xticklabels(['None'] + list('Vol_{}'.format(v) for v in types_cv) + ['Multi'], rotation=90)
        ax.contour(F,alpha=0.5)
        plt.suptitle( (savename + '\n' 
               + 'Dim {} | nVol {}'.format( dim,Nv) + '\n' 
               + 'ExpSig {:.3e} | VolSig {:.3e} | bandwidth {:.2f}'.format(esig,vsig,kFactor) ) )        
        plt.savefig(name)
        plt.close()                
                        
        name = './figures/' + savename + '_ProportionLabeledStates_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])
        ax = plt.subplot(111)                        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width(), height),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        labels = goodStates[1:]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        e_frac_full = list(barChart_full[e]/np.sum(CE==e) for e in types_ce)
        e_frac = list(barChart[e]/np.sum(CE==e) for e in types_ce)
        rects1 = ax.bar(x , e_frac_full, width, color='r')
        rects2 = ax.bar(x , e_frac, width, color='b')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Within-cluster points (fraction)')
        #ax.set_title(( savename + 'Dim {} | nVol {}'.format( dim,Nv) ))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)        
        autolabel(rects1)
        autolabel(rects2)
        #fig.tight_layout()
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()
        
        name = './figures/' + savename + '_TimeAlignedSeperability_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[11,8.5])#[3.5,2.5])
        ax = plt.subplot(111)
        labels = list( '{}, mean {:.3f}'.format(a,b) for a,b in zip(goodStates[1:], e_frac))
        xs = np.arange(max(CT))
        et_array = np.zeros([len(xs),5])
        etf_array = np.zeros([len(xs),5])
        for ei, e in enumerate(types_ce):
            sampt = np.array([CT[tm] for tm in list(lineDraw[e])])
            samptf = np.array([CT[tm] for tm in list(lineDraw_full[e])])
            popt = CT[CE==e]
            for t in xs:
                ppt = np.sum(popt==t)
                sst = np.sum(sampt==t)
                sstf = np.sum(samptf==t)
                et_array[int(t),int(ei)] = sst/ppt 
                etf_array[int(t),int(ei)] = sstf/ppt 
        lines1 = ax.plot(et_array)
        lines2 = ax.plot(etf_array,'-.',alpha=0.3)
        for line1, line2 in zip(lines1,lines2):
            line2.set_color(line1.get_color())
        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('Within-cluster points (fraction)')
        #ax.set_title(( savename + 'Dim {} | nVol {}'.format( dim,Nv) )        ax.set_title(dim)
        #ax.set_title('Dim {}'.format(dim))
        plt.legend(labels)
        ax.set_xlabel('Image #, block-design experiment')
        #fig.tight_layout()
        plt.suptitle(name)                
        plt.savefig(name)
        plt.close()
        
        name = './figures/' + savename + '_Specificity_Dim' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])        
        ax = plt.subplot(211)                
        plt.imshow(wtr)
        plt.colorbar()       
        ax = plt.subplot(2,1,2)                
        scts1 = plt.plot(boxPlot_all[:,0],boxPlot_all[:,1:],'*')
        plt.legend(goodStates[1:])
        scts2 = plt.plot(boxPlot[:,0],boxPlot[:,1:],'o',markerfacecolor=None)        
        for sct1, sct2 in zip(scts1,scts2):
            sct2.set_color(sct1.get_color())
        ax.set_ylabel('# Labeled in cluster / \n Mean # if random')
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()        
        
        name = './figures/' + savename + '_Generalizability_Dim' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])        
        ax = plt.subplot(211)        
        plt.imshow(wtr)
        plt.colorbar()        
        ax = plt.subplot(2,1,2)                
        scts1 = plt.plot(proportions_all[:,0],proportions_all[:,1:],'*')
        plt.legend(goodStates[1:])
        scts2 = plt.plot(proportions[:,0],proportions[:,1:],'o',markerfacecolor=None)        
        for sct1, sct2 in zip(scts1,scts2):
            sct2.set_color(sct1.get_color())
        ax.set_ylabel('# Labeled in cluster / # Expmt total')
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()        
        
 
    saves = {'eImgs': e_ImgOuts, 'vImgs': v_ImgOuts, 'wSaves': wSaves, 'pSaves': pSaves}    
    name = './figures/' + savename + '_saves.pkl'    
    with open(name, 'wb') as file:
        pickle.dump(saves, file, pickle.HIGHEST_PROTOCOL)
        
    name = './figures/' + savename + '_aSaves.pkl'    
    with open(name, 'wb') as file:
        pickle.dump(aSaves, file, pickle.HIGHEST_PROTOCOL)
    
def runTrainingPlots():
    Types = ['Simplex','Diagrams'] + ['Landscapes{}'.format(h) for h in range(5)]
    
    for type in Types:
        plotEmbeddingTraining(saveloc = curdir, display=False, nVols = -1, localdir = 'UMAPxyTrain{}/'.format(type), figname='TrainingUMAP{}.png'.format(type))
        
def runTestingPlots():
    Metrics = ['Diagram'] #+ ['Landscape{}'.format(h) for h in range(5)]
    Metrics = ['Simplex']
    
    for metric in Metrics:    
        #plotAggUMAPs(saveloc = curdir, figname='TestingUMAP{}.png'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type))
        #plotSSIM(saveloc = curdir, savename='TestingSSIM{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        #plotStateMap(saveloc = curdir, savename='TestingWatershed{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        plotCombinedPlots(saveloc = curdir, savename='TestingAll{}'.format(metric), display=False, nVol = -1, localdir = 'UMAPxyAll{}/'.format(metric), precalc=False)

        
def plotInformationTheory(saveloc = curdir, savename='TestingStates', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False, metric=None):        
    
    import umap
    from pyitlib import discrete_random_variable as drv

    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1
            ttime[i,0] = initt
        else:
            ttime[i,0] = 0
            inite = t
            initt = 0
    print('ttime hist')
    print(np.histogram(ttime))
    print(ttime.shape)
    print(ttime)

    trimLen = p_utils.trimLen

    embloc = saveloc + localdir
    tabloc = saveloc + 'train0table/'

    volunteers = getSecondList(embloc)

    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:

            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            #print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))

    Nv = nV = len(h3)

    Nv = len(volunteers)
    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    ce = []
    ct = []
    cv = []
    ci = []

    for ni in range(Nv):
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])
        cv.append(np.zeros(ct[-1].shape)+ni)        
        ci.append(np.arange(ct[-1].shape[0]))
        #print([ci[-1].shape)
        #print(np.histogram(ce[-1]))
        #print(np.histogram(ct[-1]))

    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)

    CE = np.concatenate(ce).ravel()
    print(['CE shape is: ', CE.shape])
    types_ce = np.unique(CE)
    # Set index boundaries for list of events
    mxe1 = np.max(types_ce)+1
    mne = 0
    
    CV = np.concatenate(cv).ravel()
    print(['CV shape is: ', CV.shape])
    types_cv = np.unique(CV)
    # set index boundaries for list of volunteers
    mxv1 = np.max(types_cv)+1
    mnv = -1

    CT = np.concatenate(ct).ravel()

    CI = np.concatenate(ci).ravel()
    types_ci = np.unique(CI)
    mxi1 = np.max(types_ci)+1
    
    entropy_outs = {}
    cluster_outs = {}
    significant_outs = {}
    
    if metric == 'Diagram':        
        name = './figures/TestingAllDiagram_aSaves.pkl'    
    elif metric == 'Simplex':
        name = './figures/TestingAllSimplex_aSaves.pkl'    
        
    with open(name, 'rb') as file:
        aSaves = pickle.load(file)
    
    #For experiments

    for dim in range(maxdim+1):

        [xx,yy,xmin,xmax,ymin,ymax] = aSaves[dim]['grid']
        F = aSaves[dim]['F']
        wtr = aSaves[dim]['wtr']                       
        uwtr = np.unique(wtr)
        #print(uwtr)
        
        data = np.concatenate([h3[nv][dim] for nv in range(Nv)])
        print('data is shape {}'.format(data.shape))
        data_vec = np.arange(data.shape[0])
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(xmin,xmax,num=len(xx)))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(ymin,ymax,num=len(yy)))
        ysnap = list(map(snap_fun,data[:,1]))
        points = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        

        entropy = np.full(len(uwtr)+1,1,dtype='float')
        
        sig = 0.05
        esig = 0.05/(len(uwtr)*len(types_ce))
        vsig = 0.05/(len(uwtr)*len(types_cv))
        
        entropy_outs[dim] = np.full(len(CV),1,dtype='float')
        cluster_outs[dim] = np.full(len(CV),0,dtype='int')        
        significant_outs[dim] = np.full(len(CV),False)        

        e_routs = pd.DataFrame(columns=types_ce,dtype=float)
        
        numEinW = np.zeros(mxe1)
        
        for w in uwtr:
            #print('Doing wtr region # ' + str(w))
            rows, cols = np.nonzero(wtr==w)
            pts = [[xx[rows[i],0], yy[0,cols[i]]] for i in range(len(rows)) ]
            obj = geo.MultiPoint(pts)
            wpolys = obj.convex_hull
            wouts = [i for i,k in enumerate(points) if k.intersects(wpolys)]        
            
            cluster_outs[dim][wouts] = w

            if not wouts:
                print('no points under wtr region # ' + str(w))
                entropy[w] = 1#np.nan                
                continue
            else:
                print('Site {} of watershed holds {} points'.format(w,len(wouts)))
                               
            rands = [np.random.choice(CE, size=len(wouts), replace=False) for _ in range(200)]
            for e in types_ce:
                dist = np.array([np.sum(rr==e) for rr in rands]).ravel()
                epts = np.array([CE[pt]==e for pt in wouts])
                wpts = np.array(wouts)[epts]
                obs = np.sum(epts)
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd                
                p_value = st.norm.sf(zstat)
                e_routs.loc[w,e] = p_value            
                
                numEinW[e] = obs             
                            
            sum_w = np.sum(numEinW)
            normalizer = np.log2(np.sum(numEinW>0))
            entropy[w] = -np.sum([obs/sum_w * np.log2(obs/sum_w) / normalizer for obs in numEinW if obs != 0])
            entropy_outs[dim][wouts] = entropy[w]
                
            if any(e_routs.loc[w,:].values < esig):                
                e_all = e_routs.columns.values[np.array(e_routs.loc[w,:].values < esig).astype('bool')]
                for ei in list(e_all):                   
                    sigInds = (cluster_outs[dim]==w) & (CE==ei)                    
                    significant_outs[dim][sigInds] = True
                    
        print(entropy)
        
        makedirs('./figures/' ,exist_ok=True)        
        
        xlines = np.diff(ce[0].ravel())
        xlines = np.arange(len(xlines))[xlines.ravel()!=0]+0.5
        xticks = np.array([0]+list(xlines)+[len(ce[0])])
        xticks = xticks[:-1]+np.diff(xticks)/2
        xtick_labels = ce[0][xticks.astype('int')]        
        xtick_labels = [goodStates[xt] for xt in np.concatenate(xtick_labels)]
        
        # For entropys
        
        # For entropys
        name = './figures/' + savename + '_entropy_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        plt.suptitle( (savename + '_entropy \n' 
                       + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        ax = plt.subplot(111) 

        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(entropy_outs[dim],[-1,len(types_cv)],order='F')
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')

        for r in range(Xm.shape[0]):
            Y, inds, counts = np.unique(Ym[r,:], return_index=True, return_counts=True)
            X = Xm[r,:len(Y)]
            c = 'b'
            s = counts**2
            e = Sigm[r,inds]
            et = e==True
            ef = e==False
            plt.scatter(X[et],Y[et],c='g',s=s[et],marker='o')            
            plt.scatter(X[ef],Y[ef],c='c',s=s[ef],marker='o')        

        [left,right] = ax.get_xlim()
        [bottom,top] = ax.get_ylim()

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        plt.plot(X,Y,'r-')
        
        # Plot also timeseries for one volunteer
        plt.plot(Xm[:,0],Ym[:,0],'k:')        

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=40)
        
        plt.savefig(name)
        plt.close()                
        
        # for cluster VI
        name = './figures/' + savename + '_clusterVI_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        plt.suptitle( (savename + '_cluster VI \n' 
                       + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        ax = plt.subplot(111) 
        
        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(cluster_outs[dim],[-1,len(types_cv)],order='F')
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')
        for r in range(Xm.shape[0]):
            Y, inds, counts = np.unique(Ym[r,:], return_index=True, return_counts=True)
            X = Xm[r,:len(Y)]
            c = 'b'
            s = counts**2
            e = Sigm[r,inds]
            et = e==True
            ef = e==False
            plt.scatter(X[et],Y[et],c='g',s=s[et],marker='o')            
            plt.scatter(X[ef],Y[ef],c='c',s=s[ef],marker='o')        

        [left,right] = ax.get_xlim()
        [bottom,top] = ax.get_ylim()

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        plt.plot(X,Y,'r-')
        
        # Plot also timeseries for one volunteer
        plt.plot(Xm[:,0],Ym[:,0],'k:')        

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=40)

        plt.savefig(name)
        plt.close()                                

        # Plot Trajectories

        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(cluster_outs[dim],[-1,len(types_cv)],order='F')    
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')

        moves = []    
        for nv in range(Nv):
            temp = h3[nv][dim]
            tempx = np.diff(h3[nv][dim][:,0])
            tempy = np.diff(h3[nv][dim][:,1])
            moves.append(np.sqrt(np.add(tempx**2,tempy**2)))

        movesA = np.array(moves)
        movesA = np.abs(movesA)
        movesM = np.mean(movesA,axis=0)
        movesS = np.std(movesA,axis=0)
        movesX = np.arange(len(movesM))+0.5

        hops = np.mean(np.diff(Ym,axis=0) == 0, axis=1)    

        sigsY = np.mean(Sigm, axis=1)
        sigsX = np.arange(len(sigsY))
        
        
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig, host = plt.subplots()
        name = './figures/' + savename + '_Trajectories_H' + str(dim) + '.png'
        plt.suptitle( (savename + '_trajectories \n' 
                   + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        fig.set_figwidth(11)
        fig.set_figheight(8.5)
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        #movesS = movesS*0

        p1, = host.plot(movesX, movesM, "b*", label="Average movement")
        #p1m, = host.plot(movesX, movesM + movesS, "b.", alpha=0.2, label="Std movement")
        p1p, = host.plot(movesM + movesS, "b.", alpha=0.2, label="Std movement")
        #p1p = host.fill_between(movesX, movesM - movesS, movesM + movesS, color='gray', alpha=0.2, label="StdDev movement")
        p2, = par1.plot(movesX, hops, "go", alpha=0.5, label="Proportion stable")
        p3, = par2.plot(sigsX, sigsY, "kx", alpha=0.5, label="Proportion w/in sig clust")

        host.set_xlim(min(sigsX), max(sigsX) )
        bottom = 0#min(np.subtract(movesM,movesS))
        top = max(np.add(movesM,movesS))
        host.set_ylim(bottom,top)
        par1.set_ylim(-0.01, 1.01)
        par2.set_ylim(-0.01, 1.01)

        host.set_xlabel("Scan TR")
        host.set_ylabel("Average movement over embedding")
        par1.set_ylabel("Proportion stable between TRs")
        par2.set_ylabel("Proportion within significant cluster")

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())

        host.set_xticks(xticks)
        host.set_xticklabels(xtick_labels, rotation=40)

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)

        lines = [p1, p1p, p2, p3]

        host.legend(lines, [l.get_label() for l in lines])

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        host.plot(X,Y,'r-')

        plt.savefig(name)
        plt.close()                                
        
    name = './figures/' + savename + '_bSaves.pkl'    
    with open(name, 'wb') as file:
        pickle.dump(cluster_outs, file, pickle.HIGHEST_PROTOCOL)
        
def runInformationTheory():
    Metrics = ['Diagram'] #+ ['Landscape{}'.format(h) for h in range(5)]
    Metrics = ['Simplex']
    
    for metric in Metrics:    
        #plotAggUMAPs(saveloc = curdir, figname='TestingUMAP{}.png'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type))
        #plotSSIM(saveloc = curdir, savename='TestingSSIM{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        #plotStateMap(saveloc = curdir, savename='TestingWatershed{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        plotInformationTheory(saveloc = curdir, savename='AllInformationTheory{}'.format(metric), display=False, nVol = -1, localdir = 'UMAPxyAll{}/'.format(metric), precalc=False, metric=metric) 
        

def plotPerformanceTesting(saveloc = curdir, savename='TestingStates', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False, metric=None, doUMAP=False):        

    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats
    
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    
    maxdim = p_utils.maxdim
    trimLen = p_utils.trimLen    
    
    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']    
    
    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    rtime = np.zeros(etime.shape)
    inite = -1
    initt = -1

    for i, t in enumerate(etime):
        if t == inite:
            initt += 1
            ttime[i,0] = initt
        else:
            ttime[i,0] = 0
            inite = t
            initt = 0

        if i < 508:
            rtime[i] = 1
        else:
            rtime[i] = 2
    print('ttime hist')
    print(np.histogram(ttime))
    print(ttime.shape)
    print(ttime)
    
    # Establish metadata labels
    embloc = saveloc + localdir

    volunteers = p_utils.getSecondList(embloc)
    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:
            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                h3.append(pickle.load(file))

    Nv = nV = len(volunteers)

    state_dict = {}
    for i , s in enumerate(goodStates):
        state_dict[i] = s        

    ce = []
    ct = []
    cr = []
    cv = []
    ci = []

    for vi, vol in enumerate(volunteers):
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])
        cr.append(rtime[trimLen:-trimLen])
        cv.append(np.zeros(ct[-1].shape)+vi)        
        ci.append(np.arange(ct[-1].shape[0]))
        #print([ci[-1].shape)
        #print(np.histogram(ce[-1]))
        #print(np.histogram(ct[-1]))        

    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)
    
    CE = np.concatenate(ce).ravel()
    print(['CE shape is: ', CE.shape])
    types_ce = np.unique(CE)
    # Set index boundaries for list of events
    mxe1 = np.max(types_ce)+1
    mne = 0

    CV = np.concatenate(cv).ravel()
    print(['CV shape is: ', CV.shape])
    types_cv = np.unique(CV)
    # set index boundaries for list of volunteers
    mxv1 = np.max(types_cv)+1
    mnv = -1

    CT = np.concatenate(ct).ravel()

    CR = np.concatenate(cr).ravel()    

    CI = np.concatenate(ci).ravel()
    types_ci = np.unique(CI)
    mxi1 = np.max(types_ci)+1
    
    with open('../performance.pkl','rb') as file:    
        performance_log = pickle.load(file)
    performance = {}
    performance_mean = {}
    for perf in performance_log:
        performance[perf] = np.full(CV.shape,np.nan,dtype='float')
        performance_mean[perf] = np.full(len(types_cv),np.nan,dtype='float')

    for ii, [vv, ee, rr] in enumerate(zip(CV, CE, CR)):
        if ee>2:        
            for perf in performance_log:
                performance[perf][ii] = performance_log[perf].loc[
                    volunteers[int(vv)][:5],goodStates[int(ee)]+'_'+str(int(rr))]  
        
    for vi in types_cv.astype(int):
        for perf in performance_log:
            performance_mean[perf][vi] = np.mean( 
                (performance_log[perf].loc[volunteers[vi][:5],:].values).astype('float') )

    if doUMAP == True:
    
        import umap
         
        if metric == 'Diagram':
            volloc = saveloc + 'ind1/'    
        elif metric == 'Simplex':
            volloc = saveloc + 'ind0/'

        # Make table of existing data
        # indices pulled from volloc as previous step (p3_....py) runs over all available volunteers
        all_times_vols = p_utils.getSecondList(volloc)
        allInds = []
        volInds = {}
        allNames = []
        allNames_dict = {}
        dict_count = -1
        dLen = 0
        for voln in tqdm(all_times_vols,desc='Loading timing info.'):
            allInds.append(np.load(volloc + str(voln) + '.npy', allow_pickle=True))
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
        for dim in range(maxdim+1):
            dataMat[dim] = np.full([lan,lan],1000.0)
            dataMat[dim][np.diag_indices(lan)] = 0

        # Load distances
        locBeg = saveloc + '{}AllDist_HN/Begun/'.format(metric)    
        makedirs(locBeg ,exist_ok=True)
        begun = p_utils.getSecondList(locBeg)            

        locFin = saveloc + '{}AllDist_HN/Finished/'.format(metric)    
        makedirs(locFin ,exist_ok=True)
        finished = p_utils.getSecondList(locFin)                            
        allFin = list(allNames_dict[fin] for fin in finished)
        for fin in tqdm(finished, desc='Loading incremental data'):
            fname = locFin + fin + '.npy'
            tempD = np.load(fname)
            fname = locBeg + fin + '.npy'
            otherInds = np.load(fname)
            for dim in range(maxdim+1):            
                dataMat[dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                dataMat[dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()                                             

        print('Make group embedding')

        embedG = {}                
        for dim in range(maxdim+1):        
            groupG = dataMat[dim][allFin,:][:,allFin]
            reducer = umap.UMAP(n_neighbors=5, n_components=9, metric=p_utils.donotripit, n_epochs=None, learning_rate=1.0, init='random', min_dist=0.1, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, verbose=False)
            embedG[dim] = reducer.fit(groupG)
            print(np.histogram(embedG[dim].embedding_))
            print(len(embedG))            
            
    entropy_outs = {}
    cluster_outs = {}
    significant_outs = {}

    e_ImgOuts = {}

    if metric == 'Diagram':        
        name = './figures/TestingAllDiagram_aSaves.pkl'    
    elif metric == 'Simplex':
        name = './figures/TestingAllSimplex_aSaves.pkl'    
    with open(name, 'rb') as file:
        aSaves = pickle.load(file)
        
    for dim in range(maxdim+1):        
        [xx,yy,xmin,xmax,ymin,ymax] = aSaves[dim]['grid']
        F = aSaves[dim]['F']
        wtr = aSaves[dim]['wtr']                       

        data = np.concatenate([h3[nv][dim] for nv in range(Nv)])
            
        print('data is shape {}'.format(data.shape))
        data_vec = np.arange(data.shape[0])
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(xmin,xmax,num=len(xx)))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(ymin,ymax,num=len(yy)))
        ysnap = list(map(snap_fun,data[:,1]))
        points = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        
        
        uwtr = np.unique(wtr)
        
        entropy = np.full(len(uwtr)+1,np.nan)

        sig = 0.05
        esig = 0.05/(len(uwtr)*len(types_ce))
        vsig = 0.05/(len(uwtr)*len(types_cv))
        psig = 0.05/(len(uwtr)*len(performance.keys()))

        entropy_outs[dim] = np.full(len(CV),1,dtype='float')
        cluster_outs[dim] = np.full(len(CV),0,dtype='int')        
        significant_outs[dim] = np.full(len(CV),False)        

        e_ImgOuts[dim] = np.zeros(wtr.shape)

        e_routs = pd.DataFrame(columns=types_ce,dtype=float)

        p_routs = {}
        for e in types_ce:
            p_routs[e] = pd.DataFrame(columns=list(performance.keys()),dtype=float)

        numEinW = np.zeros(mxe1)

        print('Set has {} clusters'.format(len(uwtr)))
        for w in uwtr:
            #print('Doing wtr region # ' + str(w))
            rows, cols = np.nonzero(wtr==w)
            pts = [[xx[rows[i],0], yy[0,cols[i]]] for i in range(len(rows)) ]
            obj = geo.MultiPoint(pts)
            wpolys = obj.convex_hull
            wouts = [i for i,k in enumerate(points) if k.intersects(wpolys)]        

            cluster_outs[dim][wouts] = w    

            if not wouts:
                print('no points under wtr region # ' + str(w))
                entropy[w] = 1#np.nan                
                continue
            else:
                print('Site {} of watershed holds {} points'.format(w,len(wouts)))

            rands = [np.random.choice(CE, size=len(wouts), replace=False) for _ in range(200)]
            for e in types_ce:
                dist = np.array([np.sum(rr==e) for rr in rands]).ravel()
                epts = np.array([CE[pt]==e for pt in wouts])
                wpts = np.array(wouts)[epts]
                obs = np.sum(epts)
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd                
                p_value = st.norm.sf(zstat)
                e_routs.loc[w,e] = p_value            

                numEinW[e] = obs             

            sum_w = np.sum(numEinW)
            normalizer = np.log2(np.sum(numEinW>0))
            entropy[w] = -np.sum([obs/sum_w * np.log2(obs/sum_w) / normalizer for obs in numEinW if obs != 0])
            entropy_outs[dim][wouts] = entropy[w]

            e_all = e_routs.columns.values[np.array(e_routs.loc[w,:].values < esig).astype('bool')]
            
            if any(e_all):                        
                e_best = e_routs.columns.values[np.argmin(e_routs.loc[w,:].values)]            
                for ei in list(e_all):                   
                    sigInds = (cluster_outs[dim]==w) & (CE==ei)                                
                    significant_outs[dim][sigInds] = True
                if np.sum(e_routs.loc[w,:].values < esig)>1:
                    e = mxe1
                else:
                    e = e_best
            else:
                 e = mne
            e_ImgOuts[dim][wtr==w] = e            
            
            for e in types_ce[2:]:
                lenw = len(CE[wouts]==e)
                if lenw > 0:
                    rands = [np.random.choice(data_vec[CE==e], size=lenw, replace=False) for _ in range(200)]            
                    for perf in performance:
                        distrabution = np.array([np.mean(performance[perf][rrs]) for rrs in rands]).ravel()
                        rmea = np.mean(distrabution)
                        rstd = np.std(distrabution)
                        epts = np.array([performance[perf][pt] for pt in wouts if CE[pt] == e])
                        obs = np.mean(epts)
                        zstat = -(rmea-obs)/rstd
                        p_value = st.norm.sf(np.abs(zstat)) * 2 
                        if p_value < psig and any(e_all==e):
                            p_routs[e].loc[w,perf] = ((-1) ** (zstat<0)) # p_value

        for e in types_ce[2:]:    
            p_routs[e].fillna(0, inplace=True)                

        print(entropy)

        # Plot statistics
        makedirs('./stats/' ,exist_ok=True)        
        
        if doUMAP:            
            h4 = []            
            for voln in all_times_vols:          
                print('Make {} embedding'.format(voln))
                Dembs = {}                    
                volFin = volInds[voln]
                group = dataMat[dim][volFin,:][:,allFin]
                print(group.shape)
                h4.append(embedG[dim].transform(group))
            data = np.concatenate([h4[nv] for nv in range(Nv)])                        
        
        for e in types_ce[2:]:
            cluster_sets =  np.full(len(CV),np.nan)
            cluster_sets_mean =  np.full(len(CV),np.nan)
            cur_dist = data[0,:]
            distance_sets = {}
            distance_sets[dim] = np.full(len(CV),np.nan)
            for ii, [vv, ee, rr] in enumerate(zip(CV, CE, CR)):
                if ee==e:        
                    cluster_sets[ii] = int(str(int(rr)) + str(int(vv)).zfill(2) + str(int(ee)).zfill(2))
                    cluster_sets_mean[ii] = int(str(int(vv)))
                    distance_sets[dim][ii] = np.sum(np.abs(np.subtract(cur_dist,data[ii+1,:]))**2)**(1/2)
                if ii<len(CV)-1:
                    cur_dist = data[ii+1,:]

            uniq = np.unique(cluster_sets[~np.isnan(cluster_sets)])
            uniq_mean = np.unique(cluster_sets_mean[~np.isnan(cluster_sets)])

            stats = {}
            stats['cluster_entropy'] = np.full(CV.shape, np.nan)
            stats['sum_distance_traveled'] = np.full(CV.shape, np.nan)
            stats['modal_step_distance'] = np.full(CV.shape, np.nan)
            stats['percent_within_significant_clusters'] = np.full(CV.shape, np.nan)
            vecs = {}
            vecs['vec_out'] = []
            vecs['vec_sets'] = []

            for uu in uniq_mean:  
                vec = cluster_sets_mean==uu
                vecs['vec'] = vec
                vecs['vec_out'].append(np.argmax(vec))
                vecs['vec_sets'].append(vec)
                ees = cluster_outs[dim][vec]
                le = len(ees)
                ue = np.unique(ees)
                normalizer = 1#np.log2(len(ue))
                ent = -np.sum([np.sum(ees==en)/le * np.log2(np.sum(ees==en)/le) / normalizer for en in ue])
                stats['cluster_entropy'][vec] = ent
                stats['sum_distance_traveled'][vec] = np.sum(distance_sets[dim][vec])
                stats['modal_step_distance'][vec] = np.median(distance_sets[dim][vec])
                stats['percent_within_significant_clusters'][vec]= np.mean(significant_outs[dim][vec])                    

            for stat_type in stats:
                name = './stats/' + savename + '_LinReg_H{}_{}_{}'.format(dim, goodStates[e], stat_type) + '.png'
                print(name)

                fig, host = plt.subplots()
                plt.suptitle( name )        
                fig.set_figwidth(11)
                fig.set_figheight(6)
                fig.subplots_adjust(right=0.75)

                par1 = host.twinx()
                par2 = host.twinx()

                par2.spines["right"].set_position(("axes", 1.2))
                make_patch_spines_invisible(par2)
                par2.spines["right"].set_visible(True)

                vec = vecs['vec_out']#~np.isnan(cluster_ents[dim])

                XX = stats[stat_type][vec]                
                
                X2 = sm.add_constant(XX)

                pStats = {}
                for perf in performance:
                    y = performance[perf][vec]
                    est = sm.OLS(y, X2)
                    est2 = est.fit()
                    pStats[perf] = est2.pvalues[1]

                p1, = host.plot(XX, performance['correct'][vec],
                                "g*", alpha=0.6, label="Percent Correct")        
                z1f = np.polyfit(XX, performance['correct'][vec],1)
                z1y = np.poly1d(z1f)
                z1s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z1f[0],z1f[1],pStats['correct'] )
                p1z, = host.plot(XX,z1y(XX), "g-", label=z1s)

                p2, = par1.plot(XX, performance['time'][vec],
                                "k^", alpha=0.6, label="Response Time")
                z2f = np.polyfit(XX, performance['time'][vec],1)
                z2y = np.poly1d(z2f)
                z2s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z2f[0],z2f[1],pStats['time'] )
                p2z, = par1.plot(XX,z2y(XX), "k-", label=z2s)

                p3, = par2.plot(XX, performance['missing'][vec],
                                "m.", alpha=0.6, label="Percent Missing")
                z3f = np.polyfit(XX, performance['missing'][vec],1)
                z3y = np.poly1d(z3f)
                z3s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z3f[0],z3f[1],pStats['missing'] )
                p3z, = par2.plot(XX,z3y(XX), "m-", label=z3s)

                if stat_type.find('distance')>0:
                    host.set_xlabel(stat_type + ' {}-D UMAP'.format(data.shape[1]))
                else:
                    host.set_xlabel(stat_type)
                host.set_ylabel("Percent Correct")
                par1.set_ylabel("Response Time")
                par2.set_ylabel("Percent Missing")

                host.yaxis.label.set_color(p1.get_color())
                par1.yaxis.label.set_color(p2.get_color())
                par2.yaxis.label.set_color(p3.get_color())

                par1.invert_yaxis()
                par2.invert_yaxis()
                '''
                host.set_xticks(xticks)
                host.set_xticklabels(xtick_labels, rotation=40)
                '''
                tkw = dict(size=4, width=1.5)
                host.tick_params(axis='y', colors=p1.get_color(), **tkw)
                par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
                par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
                host.tick_params(axis='x', **tkw)

                lines = [p1, p1z, p2, p2z, p3, p3z]

                host.legend(lines, [l.get_label() for l in lines])

                plt.savefig(name)
                plt.close()                

        for e in types_ce[2:]:
            name = './stats/' + savename + '_ClusterWise_H{}_{}'.format(dim, goodStates[e]) + '.png'
            print(name)
            
            fig = plt.figure(num = e,figsize=[8,8])    

            ax = plt.subplot(221)
            img = ax.imshow(e_ImgOuts[dim], cmap='nipy_spectral',alpha=0.5,vmin=0,vmax=mxe1)
            cbar = plt.colorbar(img, orientation='horizontal')
            cbar.set_ticks([0] + list(types_ce) + [mxe1])
            cbar.ax.set_xticklabels( goodStates + ['Multi'], rotation=90)
            ax.contour(F,alpha=0.5)               

            for i, perf in enumerate(performance,start=2):

                ax = plt.subplot(2,2,i)
                temp = wtr*0
                for w in p_routs[e].index:
                    if not np.isnan(p_routs[e].loc[w,[perf]][0]):
                        temp[wtr==w] = p_routs[e].loc[w,[perf]][0]
                img = ax.imshow(temp, cmap='coolwarm', vmin=-1, vmax=1)
                cbar = plt.colorbar(img, orientation='horizontal')
                cbar.set_label(perf,fontsize=16)        
                ax.contour(F,alpha=0.5)        

            plt.suptitle(name)
            
            plt.savefig(name)
            plt.close()                

def runPerformanceTesting():
    #Metrics = ['Diagram'] #+ ['Landscape{}'.format(h) for h in range(5)]
    Metrics = ['Simplex']
    
    for metric in Metrics:    
        #plotAggUMAPs(saveloc = curdir, figname='TestingUMAP{}.png'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type))
        #plotSSIM(saveloc = curdir, savename='TestingSSIM{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        #plotStateMap(saveloc = curdir, savename='TestingWatershed{}'.format(type), display=False, nVol = -1, localdir = 'UMAPxyTest{}/'.format(type), precalc=False)
        plotPerformanceTesting(saveloc = curdir, savename='AllPerformance{}'.format(metric), display=False, nVol = -1, localdir = 'UMAPxyAll{}/'.format(metric), precalc=False, metric=metric, doUMAP=False) 
        
def plotAllFigures(saveloc = curdir, savename='SimplexTestingStates', display=False, nVol = -1, localdir = 'UMAPxyTestSimplex/', precalc=False, doUMAP = False):        

    goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']
    
    epath = '/keilholz-lab/Jacob/TDABrains_00/data/FCStateClassif/timing.txt'
    etime = pd.read_csv(epath,sep='\t',header=0).values.astype('int')
    print('etime hist')
    print(np.histogram(etime))
    eset = np.unique(etime)
    ttime = np.zeros(etime.shape)
    rtime = np.zeros(etime.shape)
    inite = -1
    initt = -1
    for i, t in enumerate(etime):
        if t == inite:
            initt += 1
            ttime[i,0] = initt
        else:
            ttime[i,0] = 0
            inite = t
            initt = 0
        
        if i < 508:
            rtime[i] = 1
        else:
            rtime[i] = 2
    print('ttime hist')
    print(np.histogram(ttime))
    print(ttime.shape)
    print(ttime)

    trimLen = p_utils.trimLen

    # Establish metadata labels
    embloc = saveloc + localdir

    volunteers = getSecondList(embloc)
    h3 = []
    for vi, vol in enumerate(volunteers):
        if nVol == -1 or vi<nVol:

            embfile = embloc + str(vol) + '.pkl'
            with open(embfile,'rb') as file:
                #h3.append([])
                h3.append(pickle.load(file))
            print('h3[{}] is len({}) and type {}. h3[vi][0] is shape {}'.format(vi,len(h3[vi]),type(h3[vi]),h3[vi][0].shape))

    Nv = nV = len(h3)

    def setFed(data):
        x = data[:,0]
        y = data[:,1]
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)# Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:256j, ymin:ymax:256j]
        #xx, yy = np.mgrid[xmin:xmax:56j, ymin:ymax:56j]

        return xx, yy, xmin, xmax, ymin, ymax

    def getFed(data,xx=None,yy=None):
        x = data[:,0]
        y = data[:,1]

        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values,bw_method=0.08)
        kFactor = kernel.factor

        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        return f, kFactor
    
    ce = []
    ct = []
    cr = []
    cv = []
    ci = []

    for ni in range(Nv):
        ce.append(etime[trimLen:-trimLen])
        ct.append(ttime[trimLen:-trimLen])
        cr.append(rtime[trimLen:-trimLen])
        cv.append(np.zeros(ct[-1].shape)+ni)
        ci.append(np.arange(ct[-1].shape[0]))
        print(np.histogram(ce[-1]))
        print(np.histogram(ct[-1]))

    print(len(ct))
    print(ct[0].shape)
    print(np.concatenate(ct).shape)

    CE = np.concatenate(ce).ravel()
    print(['CE shape is: ', CE.shape])
    types_ce = np.unique(CE)
    # Set index boundaries for list of events
    mxe1 = np.max(types_ce)+1
    mne = 0
    
    CV = np.concatenate(cv).ravel()
    print(['CV shape is: ', CV.shape])
    types_cv = np.unique(CV)
    # set index boundaries for list of volunteers
    mxv1 = np.max(types_cv)+1
    mnv = Nv

    CT = np.concatenate(ct).ravel()

    CR = np.concatenate(cr).ravel()    

    CI = np.concatenate(ci).ravel()
    types_ci = np.unique(CI)
    mxi1 = np.max(types_ci)+1

    with open('../performance.pkl','rb') as file:    
        performance_log = pickle.load(file)

    performance = {}
    performance_mean = {}
    for perf in performance_log:
        performance[perf] = np.full(CV.shape,np.nan,dtype='float')
        performance_mean[perf] = np.full(len(types_cv),np.nan,dtype='float')

    for ii, [vv, ee, rr] in enumerate(zip(CV, CE, CR)):
        if ee>2:        
            for perf in performance_log:
                performance[perf][ii] = performance_log[perf].loc[
                    volunteers[int(vv)][:5],goodStates[int(ee)]+'_'+str(int(rr))]  
        
    for vi in types_cv.astype(int):
        for perf in performance_log:
            performance_mean[perf][vi] = np.mean( 
                (performance_log[perf].loc[volunteers[vi][:5],:].values).astype('float') )
            
    if doUMAP == True:
    
        import umap
         
        if metric == 'Diagram':
            volloc = saveloc + 'ind1/'    
        elif metric == 'Simplex':
            volloc = saveloc + 'ind0/'

        # Make table of existing data
        # indices pulled from volloc as previous step (p3_....py) runs over all available volunteers
        all_times_vols = p_utils.getSecondList(volloc)
        allInds = []
        volInds = {}
        allNames = []
        allNames_dict = {}
        dict_count = -1
        dLen = 0
        for voln in tqdm(all_times_vols,desc='Loading timing info.'):
            allInds.append(np.load(volloc + str(voln) + '.npy', allow_pickle=True))
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
        for dim in range(maxdim+1):
            dataMat[dim] = np.full([lan,lan],1000.0)
            dataMat[dim][np.diag_indices(lan)] = 0

        # Load distances
        locBeg = saveloc + '{}AllDist_HN/Begun/'.format(metric)    
        makedirs(locBeg ,exist_ok=True)
        begun = p_utils.getSecondList(locBeg)            

        locFin = saveloc + '{}AllDist_HN/Finished/'.format(metric)    
        makedirs(locFin ,exist_ok=True)
        finished = p_utils.getSecondList(locFin)                            
        allFin = list(allNames_dict[fin] for fin in finished)
        for fin in tqdm(finished, desc='Loading incremental data'):
            fname = locFin + fin + '.npy'
            tempD = np.load(fname)
            fname = locBeg + fin + '.npy'
            otherInds = np.load(fname)
            for dim in range(maxdim+1):            
                dataMat[dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                dataMat[dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()                                             

        print('Make group embedding')

        embedG = {}                
        for dim in range(maxdim+1):        
            groupG = dataMat[dim][allFin,:][:,allFin]
            reducer = umap.UMAP(n_neighbors=5, n_components=9, metric=p_utils.donotripit, n_epochs=None, learning_rate=1.0, init='random', min_dist=0.1, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, verbose=False)
            embedG[dim] = reducer.fit(groupG)
            print(np.histogram(embedG[dim].embedding_))
            print(len(embedG))            
            
    entropy_outs = {}
    cluster_outs = {}
    significant_outs = {}
    
    e_ImgOuts = {}
    v_ImgOuts = {}
    e_ImgOuts_best = {}
    v_ImgOuts_best = {}
    
    wSaves = {}
    pSaves = {}
    
    for dim in range(maxdim+1):
        
        data = np.concatenate([h3[nv][dim] for nv in range(Nv)])
        print('data is shape {}'.format(data.shape))
        data_vec = np.arange(data.shape[0])
        xx, yy, xmin, xmax, ymin, ymax = setFed(data)
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(xmin,xmax,num=len(xx)))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snap,myGrid=np.linspace(ymin,ymax,num=len(yy)))
        ysnap = list(map(snap_fun,data[:,1]))
        points = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        

        F, kFactor = getFed(data, xx=xx, yy=yy)

        wtr = watershed(1-F)
        uwtr = np.unique(wtr)

        entropy = np.full(len(uwtr)+1,1,dtype='float')
        
        sig = 0.05
        esig = 0.05/(len(uwtr)*len(types_ce))
        vsig = 0.05/(len(uwtr)*len(types_cv))
        psig = 0.05/(len(uwtr)*len(performance.keys()))

        entropy_outs[dim] = np.full(len(CV),1,dtype='float')
        cluster_outs[dim] = np.full(len(CV),0,dtype='int')        
        significant_outs[dim] = np.full(len(CV),False)        

        e_routs = pd.DataFrame(columns=types_ce,dtype=float)
        e_ImgOuts[dim] = np.zeros(wtr.shape)
        e_ImgOuts_best[dim] = np.zeros(wtr.shape)

        p_routs = {}
        for e in types_ce:
            p_routs[e] = pd.DataFrame(columns=list(performance.keys()),dtype=float)
            
        numEinW = np.zeros(mxe1)
        
        v_routs = pd.DataFrame(columns=types_cv,dtype=float)
        v_ImgOuts[dim] = -np.ones(wtr.shape)
        
        wSaves[dim] = {}
        pSaves[dim] = {}
        
        barChart = {}
        barChart_full = {}
        barChart_temp = {}
        boxPlot = np.full((max(uwtr)+1,max(types_ce)+1),np.nan).astype('float')
        boxPlot[:,0] = np.arange(max(uwtr)+1)
        boxPlot_all = np.full((max(uwtr)+1,max(types_ce)+1),0).astype('float')
        boxPlot_all[:,0] = np.arange(max(uwtr)+1)
        print(boxPlot_all.shape)
        proportions = np.full((max(uwtr)+1,max(types_ce)+1),np.nan).astype('float')
        proportions[:,0] = np.arange(max(uwtr)+1)
        proportions_all = np.full((max(uwtr)+1,max(types_ce)+1),0).astype('float')
        proportions_all[:,0] = np.arange(max(uwtr)+1)
        proportions_ref = {}
        lineDraw = {}
        lineDraw_full = {}
        lineDraw_temp = {}
        for e in types_ce:
            barChart[e] = 0            
            barChart_full[e] = 0                        
            lineDraw[e] = set()
            lineDraw_full[e] = set()
            proportions_ref[e] = np.sum(CE==e)

        print('Set has {} clusters'.format(len(uwtr)))
        for w in uwtr:
            #print('Doing wtr region # ' + str(w))
            rows, cols = np.nonzero(wtr==w)
            pts = [[xx[rows[i],0], yy[0,cols[i]]] for i in range(len(rows)) ]
            obj = geo.MultiPoint(pts)
            wpolys = obj.convex_hull
            wouts = [i for i,k in enumerate(points) if k.intersects(wpolys)]        
            wSaves[dim][w] = wouts
            
            cluster_outs[dim][wouts] = w

            if not wouts:
                print('no points under wtr region # ' + str(w))
                e_routs.loc[w,:] = np.nan 
                v_routs.loc[w,:] = np.nan 
                entropy[w] = 1#np.nan                
                continue
            else:
                print('Site {} of watershed holds {} points'.format(w,len(wouts)))
                               
            rands = [np.random.choice(CE, size=len(wouts), replace=False) for _ in range(300)]
            for e in types_ce:
                dist = np.array([np.sum(rr==e) for rr in rands]).ravel()
                epts = np.array([CE[pt]==e for pt in wouts])
                wpts = np.array(wouts)[epts]
                obs = np.sum(epts)
                rmea = np.mean(dist)
                rstd = np.std(dist)
                zstat = -(rmea-obs)/rstd                
                p_value = st.norm.sf(zstat)
                e_routs.loc[w,e] = p_value            
                barChart_temp[e] = obs
                boxPlot_all[w,e] = obs/rmea
                proportions_all[w,e] = obs/proportions_ref[e]
                lineDraw_temp[e] = np.array(data_vec[wpts])                

                numEinW[e] = obs             

            sum_w = np.sum(numEinW)
            normalizer = np.log2(np.sum(numEinW>0))
            entropy[w] = -np.sum([obs/sum_w * np.log2(obs/sum_w) / normalizer for obs in numEinW if obs != 0])
            entropy_outs[dim][wouts] = entropy[w]
                
            rands = [np.random.choice(CV, size=len(wouts), replace=False) for _ in range(200)]
            for v in types_cv:
                dist = np.array([np.sum(rr==v) for rr in rands]).ravel()
                obs = np.sum([CV[ind]==v for ind in wouts])
                rmea = np.mean(dist)
                rstd = np.std(dist)
                # For volunteers, test if not less-than mean
                zstat = (rmea-obs)/rstd 
                p_value = st.norm.sf(zstat)
                v_routs.loc[w,v] = p_value

            e_all = e_routs.columns.values[np.array(e_routs.loc[w,:].values < esig).astype('bool')]
            if any(e_all):                                
                e_best = e_routs.columns.values[np.argmin(e_routs.loc[w,:].values)]
                for ei in list(e_all):
                    boxPlot[w,ei] = boxPlot_all[w,ei]
                    barChart_full[ei] += barChart_temp[ei]                    
                    proportions[w,ei] = proportions_all[w,ei]
                    lineDraw_full[ei].update(list(lineDraw_temp[ei]))               
                    sigInds = (cluster_outs[dim]==w) & (CE==ei)                    
                    significant_outs[dim][sigInds] = True
                    
                if np.sum(e_routs.loc[w,:].values < esig)>1:
                    e = mxe1
                else:
                    e = e_best
                    barChart[e] += barChart_temp[e]                    
                    lineDraw[e].update(list(lineDraw_temp[e]))                
            else:
                e_best = e = mne
                
            e_ImgOuts[dim][wtr==w] = e
            e_ImgOuts_best[dim][wtr==w] = e_best

            # With respect to the number of points for a given experiment lying within a 
            # cluster significant for that experiment, test the hypothesis that a performance metric
            # is less than or greater than the mean performance metric for that experiment
            for e in types_ce[2:]:
                lenw = np.sum(CE[wouts]==e)
                if lenw > 0 and any(e_all==e):
                    rands = [np.random.choice(data_vec[CE==e], size=lenw, replace=False) for _ in range(300)]            
                    for perf in performance:
                        distrabution = np.array([np.mean(performance[perf][rrs]) for rrs in rands]).ravel()
                        rmea = np.mean(distrabution)
                        rstd = np.std(distrabution)
                        epts = np.array([performance[perf][pt] for pt in wouts if CE[pt] == e])
                        obs = np.mean(epts)
                        zstat = -(rmea-obs)/rstd
                        p_value = st.norm.sf(np.abs(zstat)) * 2 
                        if p_value < psig :
                            p_routs[e].loc[w,perf] = ((-1) ** (zstat<0)) # p_value -> sig less than or greater than
            
            # Testing how many volunteers appear less than expected inside cluster
            vsum = np.sum(v_routs.loc[w,:].values < vsig)
            v = Nv-vsum#mxv1            
            v_ImgOuts[dim][wtr==w] = v
            
        for e in types_ce[2:]:    
            p_routs[e].fillna(0, inplace=True)                

        print(entropy)
            
        print('For experiments, total segments = {}'.format(len(e_ImgOuts[dim].ravel())) )
        print(np.histogram(e_ImgOuts[dim]))
        print('For Volunteers, total segments = {}'.format(len(v_ImgOuts[dim].ravel())) )
        print(np.histogram(v_ImgOuts[dim]))
        
        print('bp all hist')
        print(np.histogram(boxPlot_all[:,1:]))
    
        # Do the plots
        makedirs('./figures/' ,exist_ok=True)
        
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        name = './figures/' + savename + '_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        
        ax = plt.subplot(221) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(ce[nv].ravel())
            s.append((ce[nv]!=-11).astype('int')*10)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.5,cmap='nipy_spectral',vmin=0,vmax=mxe1)
        #cnt = ax.contour(yy,-xx,F)
        ax.set_aspect('equal','box')     
        
        ax = plt.subplot(222)
        img = ax.imshow(e_ImgOuts[dim], cmap='nipy_spectral',alpha=0.5,vmin=0,vmax=mxe1)
        cbar = plt.colorbar(img, orientation='horizontal')
        cbar.set_ticks([0] + list(types_ce) + [mxe1])
        cbar.ax.set_xticklabels( goodStates + ['Multi'], rotation=90)
        ax.contour(F,alpha=0.5)        
        
        ax = plt.subplot(223) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[nv][dim][:,0])
            Y.append(h3[nv][dim][:,1])
            c.append(cv[nv])
            s.append((ct[nv]!=-11).astype('int')*10)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.5,cmap='viridis',vmin=-1,vmax=mxv1)
        #cnt = ax.contour(yy,-xx,F)
        ax.set_aspect('equal','box')            
        
        ax = plt.subplot(224)
        img = ax.imshow(v_ImgOuts[dim], cmap='viridis',alpha=0.5,vmin=-1,vmax=mxv1)
        cbar = plt.colorbar(img, orientation='horizontal')        
        cbar.ax.set_label('# not significantly few')
        #cbar.set_ticks( [-1] + list(types_cv) + [mxv1] )
        #cbar.ax.set_xticklabels(['None'] + list('Vol_{}'.format(v) for v in types_cv) + ['Multi'], rotation=90)
        ax.contour(F,alpha=0.5)
        plt.suptitle( (savename + '\n' 
               + 'Dim {} | nVol {}'.format( dim,Nv) + '\n' 
               + 'ExpSig {:.3e} | VolSig {:.3e} | bandwidth {:.2f}'.format(esig,vsig,kFactor) ) )        
        plt.savefig(name)
        plt.close()                
                        
        name = './figures/' + savename + '_ProportionLabeledStates_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])
        ax = plt.subplot(111)                        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{:.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width(), height),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        labels = goodStates[1:]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        e_frac_full = list(barChart_full[e]/np.sum(CE==e) for e in types_ce)
        e_frac = list(barChart[e]/np.sum(CE==e) for e in types_ce)
        rects1 = ax.bar(x , e_frac_full, width, color='r')
        rects2 = ax.bar(x , e_frac, width, color='b')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Within-cluster points (fraction)')
        #ax.set_title(( savename + 'Dim {} | nVol {}'.format( dim,Nv) ))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)        
        autolabel(rects1)
        autolabel(rects2)
        #fig.tight_layout()
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()
        
        name = './figures/' + savename + '_TimeAlignedSeperability_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[11,8.5])#[3.5,2.5])
        ax = plt.subplot(111)
        labels = list( '{}, mean {:.3f}'.format(a,b) for a,b in zip(goodStates[1:], e_frac))
        xs = np.arange(max(CT))
        et_array = np.zeros([len(xs),5])
        etf_array = np.zeros([len(xs),5])
        for ei, e in enumerate(types_ce):
            sampt = np.array([CT[tm] for tm in list(lineDraw[e])])
            samptf = np.array([CT[tm] for tm in list(lineDraw_full[e])])
            popt = CT[CE==e]
            for t in xs:
                ppt = np.sum(popt==t)
                sst = np.sum(sampt==t)
                sstf = np.sum(samptf==t)
                et_array[int(t),int(ei)] = sst/ppt 
                etf_array[int(t),int(ei)] = sstf/ppt 
        lines1 = ax.plot(et_array)
        lines2 = ax.plot(etf_array,'-.',alpha=0.3)
        for line1, line2 in zip(lines1,lines2):
            line2.set_color(line1.get_color())
        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('Within-cluster points (fraction)')
        #ax.set_title(( savename + 'Dim {} | nVol {}'.format( dim,Nv) )        ax.set_title(dim)
        #ax.set_title('Dim {}'.format(dim))
        plt.legend(labels)
        ax.set_xlabel('Image #, block-design experiment')
        #fig.tight_layout()
        plt.suptitle(name)                
        plt.savefig(name)
        plt.close()
        
        name = './figures/' + savename + '_Specificity_Dim' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])        
        ax = plt.subplot(211)                
        plt.imshow(wtr)
        plt.colorbar()       
        ax = plt.subplot(2,1,2)                
        scts1 = plt.plot(boxPlot_all[:,0],boxPlot_all[:,1:],'*')
        plt.legend(goodStates[1:])
        scts2 = plt.plot(boxPlot[:,0],boxPlot[:,1:],'o',markerfacecolor=None)        
        for sct1, sct2 in zip(scts1,scts2):
            sct2.set_color(sct1.get_color())
        ax.set_ylabel('# Labeled in cluster / \n Mean # if random')
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()        
        
        name = './figures/' + savename + '_Generalizability_Dim' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])#[3.5,2.5])        
        ax = plt.subplot(211)        
        plt.imshow(wtr)
        plt.colorbar()        
        ax = plt.subplot(2,1,2)                
        scts1 = plt.plot(proportions_all[:,0],proportions_all[:,1:],'*')
        plt.legend(goodStates[1:])
        scts2 = plt.plot(proportions[:,0],proportions[:,1:],'o',markerfacecolor=None)        
        for sct1, sct2 in zip(scts1,scts2):
            sct2.set_color(sct1.get_color())
        ax.set_ylabel('# Labeled in cluster / # Expmt total')
        plt.suptitle(name)        
        plt.savefig(name)
        plt.close()        
        
        xlines = np.diff(ce[0].ravel())
        xlines = np.arange(len(xlines))[xlines.ravel()!=0]+0.5
        xticks = np.array([0]+list(xlines)+[len(ce[0])])
        xticks = xticks[:-1]+np.diff(xticks)/2
        xtick_labels = ce[0][xticks.astype('int')]        
        xtick_labels = [goodStates[xt] for xt in np.concatenate(xtick_labels)]
        
        # For entropys
        name = './figures/' + savename + '_entropy_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        plt.suptitle( (savename + '_entropy \n' 
                       + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        ax = plt.subplot(111) 

        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(entropy_outs[dim],[-1,len(types_cv)],order='F')
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')

        for r in range(Xm.shape[0]):
            Y, inds, counts = np.unique(Ym[r,:], return_index=True, return_counts=True)
            X = Xm[r,:len(Y)]
            c = 'b'
            s = counts**2
            e = Sigm[r,inds]
            et = e==True
            ef = e==False
            plt.scatter(X[et],Y[et],c='g',s=s[et],marker='o')            
            plt.scatter(X[ef],Y[ef],c='c',s=s[ef],marker='o')        

        [left,right] = ax.get_xlim()
        [bottom,top] = ax.get_ylim()

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        plt.plot(X,Y,'r-')
        
        # Plot also timeseries for one volunteer
        plt.plot(Xm[:,0],Ym[:,0],'k:')        

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=40)
        
        plt.savefig(name)
        plt.close()                
        
        # for cluster VI
        name = './figures/' + savename + '_clusterVI_H' + str(dim) + '.png'
        fig = plt.figure(num = dim,figsize=[8,8])
        plt.suptitle( (savename + '_cluster VI \n' 
                       + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        ax = plt.subplot(111) 
        
        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(cluster_outs[dim],[-1,len(types_cv)],order='F')
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')
        for r in range(Xm.shape[0]):
            Y, inds, counts = np.unique(Ym[r,:], return_index=True, return_counts=True)
            X = Xm[r,:len(Y)]
            c = 'b'
            s = counts**2
            e = Sigm[r,inds]
            et = e==True
            ef = e==False
            plt.scatter(X[et],Y[et],c='g',s=s[et],marker='o')            
            plt.scatter(X[ef],Y[ef],c='c',s=s[ef],marker='o')        

        [left,right] = ax.get_xlim()
        [bottom,top] = ax.get_ylim()

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        plt.plot(X,Y,'r-')
        
        # Plot also timeseries for one volunteer
        plt.plot(Xm[:,0],Ym[:,0],'k:')        

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=40)

        plt.savefig(name)
        plt.close()                                

        # Plot Trajectories

        Xm = np.reshape(CI,[-1,len(types_cv)],order='F')
        Ym = np.reshape(cluster_outs[dim],[-1,len(types_cv)],order='F')    
        Sigm = np.reshape(significant_outs[dim],[-1,len(types_cv)],order='F')

        moves = []    
        for nv in range(Nv):
            temp = h3[nv][dim]
            tempx = np.diff(h3[nv][dim][:,0])
            tempy = np.diff(h3[nv][dim][:,1])
            moves.append(np.sqrt(np.add(tempx**2,tempy**2)))

        movesA = np.array(moves)
        movesA = np.abs(movesA)
        movesM = np.mean(movesA,axis=0)
        movesS = np.std(movesA,axis=0)
        movesX = np.arange(len(movesM))+0.5

        hops = np.mean(np.diff(Ym,axis=0) == 0, axis=1)    

        sigsY = np.mean(Sigm, axis=1)
        sigsX = np.arange(len(sigsY))
                
        # Plot trajectories
        fig, host = plt.subplots()
        name = './figures/' + savename + '_Trajectories_H' + str(dim) + '.png'
        plt.suptitle( (savename + '_trajectories \n' 
                   + 'Dim {} | nVol {}'.format( dim,Nv) ) )        
        fig.set_figwidth(11)
        fig.set_figheight(8.5)
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        #movesS = movesS*0

        p1, = host.plot(movesX, movesM, "b*", label="Average movement")
        #p1m, = host.plot(movesX, movesM + movesS, "b.", alpha=0.2, label="Std movement")
        p1p, = host.plot(movesM + movesS, "b.", alpha=0.2, label="Std movement")
        #p1p = host.fill_between(movesX, movesM - movesS, movesM + movesS, color='gray', alpha=0.2, label="StdDev movement")
        p2, = par1.plot(movesX, hops, "go", alpha=0.5, label="Proportion stable")
        p3, = par2.plot(sigsX, sigsY, "kx", alpha=0.5, label="Proportion w/in sig clust")

        host.set_xlim(min(sigsX), max(sigsX) )
        bottom = 0#min(np.subtract(movesM,movesS))
        top = max(np.add(movesM,movesS))
        host.set_ylim(bottom,top)
        par1.set_ylim(-0.01, 1.01)
        par2.set_ylim(-0.01, 1.01)

        host.set_xlabel("Scan TR")
        host.set_ylabel("Average movement over embedding")
        par1.set_ylabel("Proportion stable between TRs")
        par2.set_ylabel("Proportion within significant cluster")

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())

        host.set_xticks(xticks)
        host.set_xticklabels(xtick_labels, rotation=40)

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)

        lines = [p1, p1p, p2, p3]

        host.legend(lines, [l.get_label() for l in lines])

        X = np.tile(xlines[None,:],[2,1])
        Y = np.zeros(X.shape)
        Y[0,:] = bottom
        Y[1,:] = top
        host.plot(X,Y,'r-')

        plt.savefig(name)
        plt.close()                                
                
        if doUMAP:            
            h4 = []            
            for voln in all_times_vols:          
                print('Make {} embedding'.format(voln))
                Dembs = {}                    
                volFin = volInds[voln]
                group = dataMat[dim][volFin,:][:,allFin]
                print(group.shape)
                h4.append(embedG[dim].transform(group))
            data = np.concatenate([h4[nv] for nv in range(Nv)])                     
        
        # Do linear regression stats
        makedirs('./stats/' ,exist_ok=True)        
        for e in types_ce[2:]:
            cluster_sets =  np.full(len(CV),np.nan)
            cluster_sets_mean =  np.full(len(CV),np.nan)
            cur_dist = data[0,:]
            distance_sets = {}
            distance_sets[dim] = np.full(len(CV),np.nan)
            for ii, [vv, ee, rr] in enumerate(zip(CV, CE, CR)):
                if ee==e:        
                    cluster_sets[ii] = int(str(int(rr)) + str(int(vv)).zfill(2) + str(int(ee)).zfill(2))
                    cluster_sets_mean[ii] = int(str(int(vv)))
                    distance_sets[dim][ii] = np.sum(np.abs(np.subtract(cur_dist,data[ii+1,:]))**2)**(1/2)
                if ii<len(CV)-1:
                    cur_dist = data[ii+1,:]

            uniq = np.unique(cluster_sets[~np.isnan(cluster_sets)])
            uniq_mean = np.unique(cluster_sets_mean[~np.isnan(cluster_sets)])

            stats = {}
            stats['cluster_entropy'] = np.full(CV.shape, np.nan)
            stats['sum_distance_traveled'] = np.full(CV.shape, np.nan)
            stats['modal_step_distance'] = np.full(CV.shape, np.nan)
            stats['percent_within_significant_clusters'] = np.full(CV.shape, np.nan)
            vecs = {}
            vecs['vec_out'] = []
            vecs['vec_sets'] = []

            for uu in uniq_mean:  
                vec = cluster_sets_mean==uu
                vecs['vec'] = vec
                vecs['vec_out'].append(np.argmax(vec))
                vecs['vec_sets'].append(vec)
                ees = cluster_outs[dim][vec]
                le = len(ees)
                ue = np.unique(ees)
                normalizer = 1#np.log2(len(ue))
                ent = -np.sum([np.sum(ees==en)/le * np.log2(np.sum(ees==en)/le) / normalizer for en in ue])
                stats['cluster_entropy'][vec] = ent
                stats['sum_distance_traveled'][vec] = np.sum(distance_sets[dim][vec])
                stats['modal_step_distance'][vec] = np.median(distance_sets[dim][vec])
                stats['percent_within_significant_clusters'][vec]= np.mean(significant_outs[dim][vec])                    

            for stat_type in stats:
                name = './stats/' + savename + '_LinReg_H{}_{}_{}'.format(dim, goodStates[e], stat_type) + '.png'
                print(name)

                fig, host = plt.subplots()
                plt.suptitle( name )        
                fig.set_figwidth(11)
                fig.set_figheight(6)
                fig.subplots_adjust(right=0.75)

                par1 = host.twinx()
                par2 = host.twinx()

                par2.spines["right"].set_position(("axes", 1.2))
                make_patch_spines_invisible(par2)
                par2.spines["right"].set_visible(True)

                vec = vecs['vec_out']#~np.isnan(cluster_ents[dim])

                XX = stats[stat_type][vec]                
                
                X2 = sm.add_constant(XX)

                pStats = {}
                for perf in performance:
                    y = performance[perf][vec]
                    est = sm.OLS(y, X2)
                    est2 = est.fit()
                    pStats[perf] = est2.pvalues[1]

                p1, = host.plot(XX, performance['correct'][vec],
                                "g*", alpha=0.6, label="Percent Correct")        
                z1f = np.polyfit(XX, performance['correct'][vec],1)
                z1y = np.poly1d(z1f)
                z1s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z1f[0],z1f[1],pStats['correct'] )
                p1z, = host.plot(XX,z1y(XX), "g-", label=z1s)

                p2, = par1.plot(XX, performance['time'][vec],
                                "k^", alpha=0.6, label="Response Time")
                z2f = np.polyfit(XX, performance['time'][vec],1)
                z2y = np.poly1d(z2f)
                z2s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z2f[0],z2f[1],pStats['time'] )
                p2z, = par1.plot(XX,z2y(XX), "k-", label=z2s)

                p3, = par2.plot(XX, performance['missing'][vec],
                                "m.", alpha=0.6, label="Percent Missing")
                z3f = np.polyfit(XX, performance['missing'][vec],1)
                z3y = np.poly1d(z3f)
                z3s = "y={:.3f}x+{:.3f}; p={:.3f}".format(z3f[0],z3f[1],pStats['missing'] )
                p3z, = par2.plot(XX,z3y(XX), "m-", label=z3s)

                if stat_type.find('distance')>0:
                    host.set_xlabel(stat_type + ' {}-D UMAP'.format(data.shape[1]))
                else:
                    host.set_xlabel(stat_type)
                host.set_ylabel("Percent Correct")
                par1.set_ylabel("Response Time")
                par2.set_ylabel("Percent Missing")

                host.yaxis.label.set_color(p1.get_color())
                par1.yaxis.label.set_color(p2.get_color())
                par2.yaxis.label.set_color(p3.get_color())

                par1.invert_yaxis()
                par2.invert_yaxis()
                '''
                host.set_xticks(xticks)
                host.set_xticklabels(xtick_labels, rotation=40)
                '''
                tkw = dict(size=4, width=1.5)
                host.tick_params(axis='y', colors=p1.get_color(), **tkw)
                par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
                par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
                host.tick_params(axis='x', **tkw)

                lines = [p1, p1z, p2, p2z, p3, p3z]

                host.legend(lines, [l.get_label() for l in lines])

                plt.savefig(name)
                plt.close()                

        # Do image significance plots
        for e in types_ce[2:]:
            name = './figures/' + savename + '_ClusterWise_H{}_{}'.format(dim, goodStates[e]) + '.png'
            print(name)
            
            fig = plt.figure(num = e,figsize=[8,8])    

            ax = plt.subplot(221)
            img = ax.imshow(e_ImgOuts[dim], cmap='nipy_spectral',alpha=0.5,vmin=0,vmax=mxe1)
            cbar = plt.colorbar(img, orientation='horizontal')
            cbar.set_ticks([0] + list(types_ce) + [mxe1])
            cbar.ax.set_xticklabels( goodStates + ['Multi'], rotation=90)
            ax.contour(F,alpha=0.5)               

            for i, perf in enumerate(performance,start=2):

                ax = plt.subplot(2,2,i)
                temp = wtr*0
                for w in p_routs[e].index:
                    if not np.isnan(p_routs[e].loc[w,[perf]][0]):
                        temp[wtr==w] = p_routs[e].loc[w,[perf]][0]
                img = ax.imshow(temp, cmap='coolwarm', vmin=-1, vmax=1)
                cbar = plt.colorbar(img, orientation='horizontal')
                cbar.set_label(perf,fontsize=16)        
                ax.contour(F,alpha=0.5)        

            plt.suptitle(name)
            
            plt.savefig(name)
            plt.close()                

        
def runAllFigures():
    Metrics = ['Diagram'] #+ ['Landscape{}'.format(h) for h in range(5)]
    Metrics = ['Simplex']
    
    for metric in Metrics:    
        plotAllFigures(saveloc = curdir, savename='AllFigures{}'.format(metric), display=False, nVol = -1, localdir = 'UMAPxyAll{}/'.format(metric), precalc=False)
