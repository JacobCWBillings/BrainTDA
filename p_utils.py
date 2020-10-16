from os.path import join as OSjoin
from os.path import basename, splitext
import pathlib

import nibabel as nib
import numpy as np
import pycwt
import pywt
from pycwt.helpers import fft, fft_kwargs, rect
from scipy.signal import convolve2d
from scipy.spatial.distance import squareform, cdist
import scipy.sparse as sp
from persim import sliced_wasserstein as slw
from persim import bottleneck
from ripser import ripser
from numba import jit
import itertools
import networkx as nx
from pyitlib import discrete_random_variable as drv
from gudhi.representations import Landscape
import bisect

# Run parameters
maxdim = 1 
truncDim = 0 
TR = 1.5
knn = 333-5 

# adjust data stact
buffLen = int(450)
mixLen = int(300)
trimLen = 120 
trimLen0 = int(350)        
trimLen1 = 120
trimEdges = [124, 894]

# cwt parameters
mother = pycwt.Morlet(6)
s0 = 6.5
dj = .32
J = 14
s1 = s0 * 2**(J * dj)

frq_edges = [2,13]

#What metrics to pursue
metrics = ['simplex']

# thresholds
TARGS = np.array((0.05,0.1,0.15,0.2))
TARGS_CEIL = 1

# for persistence landscape
L_resolution = 1000
L_numLandscapes = 5 
L_fun = Landscape(resolution=L_resolution, num_landscapes=L_numLandscapes)

missingVoxels = [133, 296, 299, 302, 304]

def snapVal(myValue, myGrid=None):
    ix = bisect.bisect_right(myGrid, myValue)
    if ix == 0:
        return myGrid[0]
    elif ix == len(myGrid):
        return myGrid[-1]
    else:
        return min(myGrid[ix - 1], myGrid[ix], key=lambda gridValue: abs(gridValue - myValue))
    
def snapInd(myValue, myGrid=None):
    ix = bisect.bisect_right(myGrid, myValue)
    if ix == 0:
        return ix
    elif ix == len(myGrid):
        return ix-1
    else:
        return ix - 1 + np.argmin(np.abs(np.array([myGrid[ix-1], myGrid[ix]]) - myValue))

def importHCP(dpath, dname, flip=True, type0='float'):
    fname = OSjoin(dpath,dname)
    img = nib.load(fname)
    print('For filename: ' + str(fname) + '\n' + 'img shape is: ' + str(img.shape))
    D = img.get_fdata(dtype=type0) #projection
    if flip:
        D = D.T+0.0
    print('Output shape is ' + str(D.shape))    
    return D

def centerScan(X):
    X = X.T
    X = X-np.mean(X,axis=0)
    X = X/np.std(X,axis=0)
    return X.T

def scaleScan(X):
    X = X.T
    X = X/np.std(X,axis=0)
    return X.T

def getCWT_auxInfo(signal):    
    
    [_, sj, freqs, coi, _, _] = pycwt.cwt(signal, TR, s0=s0, dj=dj, J=J, wavelet=mother)

    return sj, freqs, coi

def getCWT_auxInfoPlus(signal):    
    
    [_, sj, freqs, coi, _, _] = pycwt.cwt(signal, TR, s0=s0, dj=dj, J=J, wavelet=mother)

    return sj, freqs, coi, mother, s0, dj, J, s1, TR

def getCWT(signal):    
    
    [coefficients, _, _, _, _, _ ] = pycwt.cwt(signal, TR, s0=s0, dj=dj, J=J, wavelet=mother)
    power = np.absolute(coefficients)

    return power

def getCWT_coeff(signal):    
    
    [coefficients, _, _, _, _, _] = pycwt.cwt(signal, TR, s0=s0, dj=dj, J=J, wavelet=mother)

    return coefficients

def cwt2smooth(W, sj=[10,100]):

    scales = np.ones([1, W.shape[1]]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S = mother.smooth(np.abs(W) ** 2 / scales, TR, dj, sj)
    
    return S

def cwt2smooth_local(WW,n=None,F=None, win=None):            
    smooth = fft.ifft(F * fft.fft(WW, axis=1, **fft_kwargs(WW[0, :])),
                      axis=1,  # Along Fourier frequencies
                      **fft_kwargs(WW[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT
    T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"            

    return T.real

def wct2Stack(ins):
    
    outs = 1-np.mean(ins[1].ravel()) # Coherence distance is 1- mean coherence
    
    return [ins[0],outs]

def coher(ins,n=None,F=None,win=None,scales1=None):            
    
    #wct = pool.map(lambda x: np.abs( mapFun( W[x[0]]*W[x[1]].conj()/ scales1 ) )**2/ (S[x[0]]*S[x[1]]), list(tqdm(itertools.combinations(samps,2))))
    
    [W1,W2,S1,S2] = ins

    W12 = W1*W2.conj()
    smooth = fft.ifft(F * fft.fft(W12 / scales1, axis=1, **fft_kwargs(W12[0, :])),
      axis=1,  # Along Fourier frequencies
      **fft_kwargs(W12[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT
    S12 = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"            
    WCT = np.abs( S12 )**2/(S1*S2)    

    return WCT

def cross(ins):            

    [W1,W2] = ins

    W12 = W1*W2.conj()

    return W12

   
def coher_postCross(ins,n=None,F=None,win=None,scales1=None):            

    #wct = pool.map(lambda x: np.abs( mapFun( W[x[0]]*W[x[1]].conj()/ scales1 ) )**2/ (S[x[0]]*S[x[1]]), list(tqdm(itertools.combinations(samps,2))))

    [W12,S1,S2] = ins

    smooth = fft.ifft(F * fft.fft(W12 / scales1, axis=1, **fft_kwargs(W12[0, :])),
      axis=1,  # Along Fourier frequencies
      **fft_kwargs(W12[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT
    S12 = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"            
    WCT = np.abs( S12 )**2/(S1*S2)

    return WCT

def crossNorms(w12, frq_edges=None):
    w12 = np.abs(w12)
    [nW, nT] = w12.shape
    nw12 = np.zeros(w12.shape)
    for t in range(nT):
        temp = w12[frq_edges[0]:frq_edges[1],t]
        nw12[frq_edges[0]:frq_edges[1],t] = temp/np.sum(temp)

    return nw12
 
def cwt2wct(W1, W2, S1, S2, sj):
        
    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, W1.shape[1]]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, TR, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    #aWCT = np.angle(W12)

    return WCT#, aWCT

def getDirs(adir):
    Dir = pathlib.Path(adir)
   
    contents = []
    for item in Dir.iterdir():
        if item.is_dir():
            contents.append(item.name)
    return contents

def getFiles(adir,prefix=''):
    Dir = pathlib.Path(adir)
    contents = []
    for item in Dir.iterdir():
        if item.is_file() and item.name.find(prefix)==0:        
            contents.append(item.name)
        
    return contents

def getFName(string):
    return splitext(basename(string))[0]

def getLists(vloc = '.'):
    #volunteers = ['102008',  '102311',  '119833',  '284646',  '786569']
    #volunteers = ['105923' , '103818', '111312']
    #volunteers = ['103818',  '105923',  '111312',  '114823',  '115320',  '125525',  '130518',  '135528',  '137128']
    #volunteers = ['114823',  '115320',  '125525',  '130518',  '135528',  '137128']

    volunteers = getFiles(vloc)
    print(volunteers)
    
    restDirs = ['rfMRI_REST1_LR',
                'rfMRI_REST1_RL',
                'rfMRI_REST2_LR',
                'rfMRI_REST2_RL']

    taskDirs = ['tfMRI_EMOTION_LR',
    'tfMRI_EMOTION_RL',
    'tfMRI_GAMBLING_LR',
    'tfMRI_GAMBLING_RL',
    'tfMRI_LANGUAGE_LR',
    'tfMRI_LANGUAGE_RL',
    'tfMRI_MOTOR_LR',
    'tfMRI_MOTOR_RL',
    'tfMRI_RELATIONAL_LR',
    'tfMRI_RELATIONAL_RL',
    'tfMRI_SOCIAL_LR',
    'tfMRI_SOCIAL_RL',
    'tfMRI_WM_LR',
    'tfMRI_WM_RL']

    taskDirs = ['tfMRI_WM_LR','tfMRI_WM_RL','tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL','tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
    #taskDirs = ['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
    #taskDirs = ['tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL']

    EVtxt = ['tfMRI_EMOTION_LR/EVs/fear.txt',
    'tfMRI_EMOTION_LR/EVs/neut.txt',
    'tfMRI_EMOTION_RL/EVs/fear.txt',
    'tfMRI_EMOTION_RL/EVs/neut.txt',
    'tfMRI_GAMBLING_LR/EVs/loss.txt',
    'tfMRI_GAMBLING_LR/EVs/win.txt',
    'tfMRI_GAMBLING_RL/EVs/loss.txt',
    'tfMRI_GAMBLING_RL/EVs/win.txt',
    'tfMRI_LANGUAGE_LR/EVs/math.txt',
    'tfMRI_LANGUAGE_LR/EVs/story.txt',
    'tfMRI_LANGUAGE_RL/EVs/math.txt',
    'tfMRI_LANGUAGE_RL/EVs/story.txt',
    'tfMRI_MOTOR_LR/EVs/lf.txt',
    'tfMRI_MOTOR_LR/EVs/lh.txt',
    'tfMRI_MOTOR_LR/EVs/rf.txt',
    'tfMRI_MOTOR_LR/EVs/rh.txt',
    'tfMRI_MOTOR_LR/EVs/t.txt',
    'tfMRI_MOTOR_RL/EVs/lf.txt',
    'tfMRI_MOTOR_RL/EVs/lh.txt',
    'tfMRI_MOTOR_RL/EVs/rf.txt',
    'tfMRI_MOTOR_RL/EVs/rh.txt',
    'tfMRI_MOTOR_RL/EVs/t.txt',
    'tfMRI_RELATIONAL_LR/EVs/match.txt',
    'tfMRI_RELATIONAL_LR/EVs/relation.txt',
    'tfMRI_RELATIONAL_RL/EVs/match.txt',
    'tfMRI_RELATIONAL_RL/EVs/relation.txt',
    'tfMRI_SOCIAL_LR/EVs/mental.txt',
    'tfMRI_SOCIAL_LR/EVs/rnd.txt',
    'tfMRI_SOCIAL_RL/EVs/mental.txt',
    'tfMRI_SOCIAL_RL/EVs/rnd.txt',
    'tfMRI_WM_LR/EVs/0bk_body.txt',
    'tfMRI_WM_LR/EVs/0bk_faces.txt',
    'tfMRI_WM_LR/EVs/0bk_places.txt',
    'tfMRI_WM_LR/EVs/0bk_tools.txt',
    'tfMRI_WM_LR/EVs/2bk_body.txt',
    'tfMRI_WM_LR/EVs/2bk_faces.txt',
    'tfMRI_WM_LR/EVs/2bk_places.txt',
    'tfMRI_WM_LR/EVs/2bk_tools.txt',
    'tfMRI_WM_RL/EVs/0bk_body.txt',
    'tfMRI_WM_RL/EVs/0bk_faces.txt',
    'tfMRI_WM_RL/EVs/0bk_places.txt',
    'tfMRI_WM_RL/EVs/0bk_tools.txt',
    'tfMRI_WM_RL/EVs/2bk_body.txt',
    'tfMRI_WM_RL/EVs/2bk_faces.txt',
    'tfMRI_WM_RL/EVs/2bk_places.txt',
    'tfMRI_WM_RL/EVs/2bk_tools.txt']

    # Don't  modify!! indices alligned to blockstxt
    Blocks = ['fear',
    'neut',
    'loss',
    'win',
    'math',
    'story',
    'lf',
    'lh',
    'rf',
    'rh',
    't',
    'match',
    'relation',
    'mental',
    'rnd',
    '0bk_body',
    '0bk_faces',
    '0bk_places',
    '0bk_tools',
    '2bk_body',
    '2bk_faces',
    '2bk_places',
    '2bk_tools',
    'rest']

    Blockstxt = ['fear.txt',
    'neut.txt',
    'loss.txt',
    'win.txt',
    'math.txt',
    'story.txt',
    'lf.txt',
    'lh.txt',
    'rf.txt',
    'rh.txt',
    't.txt',
    'match.txt',
    'relation.txt',
    'mental.txt',
    'rnd.txt',
    '0bk_body.txt',
    '0bk_faces.txt',
    '0bk_places.txt',
    '0bk_tools.txt',
    '2bk_body.txt',
    '2bk_faces.txt',
    '2bk_places.txt',
    '2bk_tools.txt']

    return volunteers, restDirs, taskDirs, EVtxt, Blocks, Blockstxt

_, restDirs, taskDirs, _, Blocks, _ = getLists(vloc='.')
goodStates = Blocks
goodStates = ['lf',
    'lh',
    'rf',
    'rh',
    't',
    'mental',
    'rnd',
    '0bk_body',
    '0bk_faces',
    '0bk_places',
    '0bk_tools',
    '2bk_body',
    '2bk_faces',
    '2bk_places',
    '2bk_tools',
    'rest']

goodRepetitions = restDirs + taskDirs


def getSecondList(vloc = '.',prefix=''):
    
    volunteers = getFiles(vloc, prefix=prefix)
        
    return [vol[:-4] for vol in volunteers]


## Distance metric calcs

@jit
def ripit(s1,s2):    
    r1 = ripser(squareform(s1),distance_matrix=True)
    r2 = ripser(squareform(s2),distance_matrix=True)
    RR = slw(r1['dgms'][1],r2['dgms'][1],M=20)
    return RR

@jit
def donotripit(s1,s2):    
    pass

def getMultiscaleIndexer(nP, nW):
    # Lookup table translating all possible nd simplices into ordinal labels
    # Possible simplices are limited by disconnected construction of the multi-layer graph
    # Goal is to store boolean sparse matrices when indexed simplex is present 
    global indexer
    indexer = {}
    i0 = 0
    for dim in range(truncDim,maxdim+1,1):
        for w in range(nW):
            vec = range(nP*w,nP*(w+1),1)
            for i, s in enumerate(itertools.combinations(vec,dim+1), start=i0):
                indexer[frozenset(np.sort(s))] = i
            i0 = i+1

def getMultiscaleIndexerRanges(nP, nW):
    # Lookup table translating all possible nd simplices into ordinal labels
    # Possible simplices are limited by disconnected construction of the multi-layer graph
    # Goal is to store boolean sparse matrices when indexed simplex is present
    global indexerRanges, indexer0
    indexerRanges = {}
    i0 = 0
    i00 = 0
    for dim in range(truncDim,maxdim+1,1):
        for w in range(nW):
            vec = range(nP*w,nP*(w+1),1)
            for i, s in enumerate(itertools.combinations(vec,dim+1), start=i0):
                pass
            i0 = i+1
        indexerRanges[dim] = [i00,i0]
        i00 = i0
    print(indexerRanges)
    indexer0 = np.zeros((1,i+1))

def getAllMulti(nP, nW):
    getMultiscaleIndexer(nP,nW)
    getMultiscaleIndexerRanges(nP,nW)

def graph2simplex(M, threshold):
    #graph2simplex lists a simplices in a graph given some threshold    
    
    # M is an MxM undirected weighted graph
     
    # Perform thresholding
    Mt = M.copy()#nx.convert_matrix.to_numpy_matrix(M)
    cutoff = threshold
    #print('cutoff = ' + str(cutoff))
    Mt[M>cutoff] = 0
    Mt[M<=cutoff] = 1

    #print(Mt)
    #print(Mt[:5,:5])

    #print('build initial graph')
    Mx=nx.from_numpy_matrix(Mt.astype('bool'))
    
    #print('graph built')
    
    temp = [np.asarray(sorted(l)) for l in nx.find_cliques(Mx)]
    mx_cliques = {}
    for clq in temp:
        if len(clq)>maxdim+1:
            for vv in itertools.combinations(clq,maxdim+1):
                mx_cliques[frozenset(vv)] = vv
        else:
            mx_cliques[frozenset(clq)] = clq
    ln_mx_cliques = len(mx_cliques)

    #print('printing sz_max_cliques_str {}'.format(sz_mx_cliques_str))
    
    #print('Max cliques Found')
    
    # initialize the simplex set
    simplex = np.zeros((1,len(indexer)),dtype='float')   

    # assign minimimum edge distance for each nodes
    some_list = list(indexer[frozenset([u])] for u in range(M.shape[0]))
    node_connect = list(min(M[u,M[u,:]>0]) for u in range(M.shape[0]))
    simplex[0,some_list] = node_connect

    def deeperSimplex(u2, buff2):
        v2 = list(u2)
        sz = len(v2)
        for vv in itertools.combinations(v2,sz-1):
            szv = len(vv)
            uu = frozenset(vv)
            maxEdge = max(list( M[a,b] for a,b in itertools.combinations(vv,2) ))
            buff2.append([uu, maxEdge])
            if szv>2:
                buff2 = deeperSimplex(uu,buff2)
        return buff2
        
    # create edges in the Hasse graph (diagram)
    # Initialize Hess    
    for u in mx_cliques:
        nodes = list(u)
        sz = len(nodes)
        if sz>2:
            buff = []
            maxEdge = max(list( M[a,b] for a,b in itertools.combinations(nodes,2) ))
            buff.append([u, maxEdge])
            buff = deeperSimplex(u,buff)
            for bu in buff:
                simplex[0, indexer[bu[0]] ] = bu[1]
        elif sz==2:
            maxEdge =  M[nodes[0],nodes[1]]
            simplex[0, indexer[u] ] = maxEdge

    
    print(simplex)
    print(simplex.data)
    print('Finished edges')
    return simplex

def calcSimplices(inputs):
    # g is a graph

    # output are labeled n-dimensional connected components
    # found after thresholding g at target values

    TT = inputs[0]
    print('Start tt {}'.format(TT))
    g = inputs[1]

    H = graph2simplex(g,TARGS_CEIL)
    G = sp.csr_matrix(H)        
    print('nnz for input {} and target {} is {}. out shape is {}:'.format(TT,TARGS_CEIL,G.nnz,G.shape))
    print('****** Finish tt {}'.format(TT))

    return G

def calcWeightedJaccard(inputs):
    A = inputs[0].toarray()
    B = inputs[1].toarray()
    
    dist = []
    Ap = A>0
    Bp = B>0
    oor = (Ap | Bp).astype('bool')
    AA = indexer0.copy()
    BB = indexer0.copy()
    AA[Ap] = TARGS_CEIL-A[Ap]
    BB[Bp] = TARGS_CEIL-B[Bp]
    #print(np.sum(oor))
    for dim in range(maxdim+1):
        vec = oor.copy()
        #print([vec.shape,indexerRanges[dim]])
        vec[0,:indexerRanges[dim][0]] = False
        vec[0,indexerRanges[dim][1]:] = False
        #print([dim,np.sum(vec)])
        numerator = np.min([AA[vec],BB[vec]],axis=0)
        denomerator = np.max([AA[vec],BB[vec]],axis=0)
        #print([numerator,numerator.shape])
        
        dist.append(1-(np.sum(numerator)/np.sum(denomerator)))

    #print(dist)
        
    return dist

def calcLandscapeDistance(inputs):
    A = inputs[0]
    B = inputs[1]

    dist = []
    for dim in range(maxdim+1):
        dist.append([])
        for num in range(L_numLandscapes):
            vec = np.arange(num*L_resolution,(num+1)*L_resolution)
            #dist.append(np.sum(np.abs(A-B)))
            dist[-1].append(np.sum(np.abs(A[dim][vec]-B[dim][vec])))
        
    return dist

def slw2(indata, normalize = False):
    outs = []
    for dim in range(maxdim+1):
        s1 = indata[0][dim]
        s2 = indata[1][dim]
        if normalize:
            s1 /= np.sum(np.diff(s1,axis=1))
            s2 /= np.sum(np.diff(s2,axis=1))
        #print(['\n',s1,s2,'\n'])
        outs.append( slw(s1,s2,M=20) )
    return outs
 
def getRipped(indata, distance_matrix=True, maxdim=1, makeSquare = True, fixInf=False, threshold=0.999999, do_cocycles=False, doLandscape=False):
    
    if makeSquare:
        A = squareform(indata)
    else:
        A = indata
    R = ripser(A, distance_matrix=distance_matrix, maxdim=maxdim, thresh = threshold, do_cocycles=do_cocycles)
    
    if fixInf:
        for r in range(len(R['dgms'])):
            rr = R['dgms'][r]
            irr = rr[:,1]==np.inf
            #mrr = np.max(rr[~irr,:].ravel())
            mrr = threshold
            #print('Number of inf lifetimes for dim {} is {}.'.format(r,np.sum(irr)))
            R['dgms'][r][irr,1] = mrr

    if doLandscape:
        R['landscape'] = {}
        for dim in range(maxdim+1):        
            if len(R['dgms'][dim])>0:
                L = L_fun.fit_transform([R['dgms'][dim]])[0]
            else:
                L = np.zeros(L.shape)
            R['landscape'][dim] = L

    return R
        
def doAgg(voln, adf):

    volunteers, restDirs, taskDirs, EVtxt, Blocks, Blockstxt = getLists(vloc='')
    
    print('adf is shape {}'.format(adf.shape))
    print('Num possible picks is {}'.format(np.sum(adf.loc[:,'pick']>-1)))
    print(np.histogram(adf.loc[:,'pick'].values))

    print('max adf ind {}'.format(np.max(adf.index.values)))

    # Look for which EVtxt states are in goodStates
    pickInds = []
    setgs = set(goodStates)
    for bi , bs in enumerate(Blocks):
        if bs in setgs:
            print([bi,bs])
            pickInds.append(adf.index.values[ (adf.loc[:,'pick']==bi).values])
            pickInds[-1] = np.random.choice(pickInds[-1],size=2, replace=False)
    inds = np.array(pickInds).ravel()
    print('max inds is {}'.format(np.max(inds)))

    print(np.histogram(np.array(inds)))
    print('Inds histogram voln {}, found {} inds'.format(voln, len(inds)))

    return inds
    
def getFlatBrain():
    # load data
    with open('flatBrainMasks.pkl','rb')as file:
        masks = pickle.load(file)

    # draw all polygons
    polyset = {}
    sides = ['L','R']

    ylim = [0,0]
    xlim = [0,0]

    for side in [1,2]:    
        sideMask = masks['brainlocs']==side
        segs = np.unique(masks['gordon'][sideMask])
        segs = segs[~np.isnan(segs)]
        segs = segs[segs>0].astype('int')
        for seg in segs:
            polyset[seg] = []
            #asegment = masks['gordon']==seg
            #asegmentside = asegment[sideMask]
            #polyset[sides[side-1]][seg]['points'] = masks['vert'+sides[side]][asegmentside,:2]
            #polyset[sides[side-1]][seg]['point_inds'] = set(np.arange(len(asegmentside))[asegmentside])    
            #polyset[sides[side-1]][seg]['polys'] = []

        segsArray = masks['gordon'][sideMask].astype('int')
        faces = masks['face'+sides[side-1]]
        for face in faces:
            seg = np.unique([segsArray[i-1] for i in face])
            kk = len(seg)
            if kk==1 and not any(seg<1):
                seg = seg[0]
                temp = np.array([masks['vert'+sides[side-1]][ff,:2] for ff in face])
                if not any(np.all(temp==0,axis=1)) and not any(pdist(temp)>10):
                    polyset[seg].append(temp)

                ylim[0] = min(ylim[0],min(temp[:,1]))
                ylim[1] = max(ylim[1],max(temp[:,1]))
                xlim[0] = min(xlim[0],min(temp[:,0]))
                xlim[1] = max(xlim[1],max(temp[:,0]))

    return masks, polyset, ylim, xlim