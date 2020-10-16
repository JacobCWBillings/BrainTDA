import pickle
from os import makedirs as makedirs
from os.path import join as OSjoin
from os.path import isfile as OSisfile
from os import chdir, getcwd
import subprocess
import random
from time import sleep

import numpy as np
import pandas as pd
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FFMpegWriter
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import squareform, cdist, pdist
from functools import partial
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import watershed
import shapely.geometry as geo
from shapely.ops import snap
import scipy.stats as st
from sklearn import datasets, linear_model, cluster
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm import tqdm
from pyitlib import discrete_random_variable as drv
import datetime as dt
import umap

import p_utils

# Static parameters

curdir = './results/'

buffLen = p_utils.buffLen
truncDim = p_utils.truncDim	
trimLen = p_utils.trimLen

goodStates = ['Null','Fixation','Rest', 'Memory', 'Video', 'Math']

saveloc = './results/'
savename = 'PaperFigures'

sig = 0.05

metrics = {'Diagram':'Diagram', 'Simplex':'Simplex', 'Strength': 'Strength', 'Activity': 'Activity'}
basedir = {}
inputsdir = {}
localdir = {}
maxdims = {}
rundims = {}
p3Name = {}
for  metric in metrics:
    if metric == 'Diagram':
        basedir[metric] = '../z18_GonzCast_WeightedCWT/'
        inputsdir[metric] = '../z18_GonzCast_WeightedCWT/results/rippedTesting/'
        #basedir[metric] = '../z22_GonzCast_WeightedCWT_Thresheld/'
        #inputsdir[metric] = '../z22_GonzCast_WeightedCWT_Thresheld/results/rippedTesting/'
        #basedir[metric] = '../z26_GonzCast_FullCWT/'
        #inputsdir[metric] = '../z26_GonzCast_FullCWT/results/rippedTesting/'
        maxdims[metric] = 2
        rundims[metric] = [0,1,2]
        p3Name[metric] = 'p8_genEmbeddings.py'
    elif metric == 'Simplex':
        basedir[metric] = '../z19_GonzCast_WeightedCWT_Simplex/'
        inputsdir[metric] = '../z19_GonzCast_WeightedCWT_Simplex/results/simplexTraining/'    
        #basedir[metric] = '../z20_GonzCast_WeightedCWT_Simplex_Thresheld/'
        #inputsdir[metric] = '../z20_GonzCast_WeightedCWT_Simplex_Thresheld/results/simplexTraining/'    
        maxdims[metric] = 1
        rundims[metric] = [1]
        p3Name[metric] = 'p3_genEmbeddings.py'
    if metric == 'Strength':
        basedir[metric] = '../z23_GonzCast_WeightedCWT_Strength/'        
        inputsdir[metric] = '../z23_GonzCast_WeightedCWT_Strength/results/strengthTraining/'
        maxdims[metric] = 0
        rundims[metric] = [0]
        p3Name[metric] = 'p3_genEmbeddings.py'
    if metric == 'Activity':
        basedir[metric] = '../z25_GonzCast_Activations/'        
        inputsdir[metric] = '../z25_GonzCast_Activations/results/activityTraining/'
        maxdims[metric] = 2
        rundims[metric] = [2]
        p3Name[metric] = 'p3_genEmbeddings.py'        
    localdir[metric] = basedir[metric] + 'results/UMAPxyAll' + metric + '/'         

conditions = {0:['Simplex', 1], 1:['Activity', 2], 2:['Strength', 0],
              3:['Diagram', 0], 4:['Diagram', 1], 5:['Diagram', 2]}
              

#conditions = {0:['Diagram', 0], 1:['Diagram', 1], 2:['Diagram', 2]}
#conditions = {0:['Activity', 0], 1:['Activity', 1], 2:['Activity', 2]}


def setFed(data):
    x = data[:,0]
    y = data[:,1]
    range1 = max([max(x) - min(x), max(y) - min(y)]) 
    deltax = (max(x) - min(x))/32
    deltay = (max(y) - min(y))/32

    xmin = min(x) - deltax
    xmax = max(x) + deltax
    ymin = min(y) - deltay
    ymax = max(y) + deltay
    
    xfrac = (xmax-xmin)/range1
    yfrac = (ymax-ymin)/range1
    
    max_grid = 256
    xsteps = int(np.round(xfrac*max_grid))
    ysteps = int(np.round(yfrac*max_grid))
    
    print(xmin, xmax, xsteps, ymin, ymax, ysteps)# Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:xsteps*1j, ymin:ymax:ysteps*1j]
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

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
        
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width(), height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

Nv = 18

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

nT = len(ct)

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
        

def getIndexerRanges(nP, nW):    
    # Lookup table translating all possible nd simplices into ordinal labels
    # Possible simplices are limited by disconnected construction of the multi-layer graph
    # Goal is to store boolean sparse matrices when indexed simplex is present
    indexerRanges = {}
    i0 = 0
    i00 = 0
    for dim in range(0,1+1,1):
        for w in range(nW):
            vec = range(nP*w,nP*(w+1),1)
            for i, s in enumerate(itertools.combinations(vec,dim+1), start=i0):
                pass
            i0 = i+1
        indexerRanges[dim] = [i00,i0]
        i00 = i0
    print(indexerRanges)
    indexerA = np.arange(i+1)
    return indexerRanges, indexerA

nP = p_utils.knn
nW = 1
    
indexerRanges, indexerA = getIndexerRanges(nP,nW)

volunteers = p_utils.getSecondList(localdir['Diagram'])

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
        
        
# Initialize data Mat            

volloc = basedir['Diagram'] + 'results/ind1/'

# Make table of existing data
# indices pulled from train_volloc as previous step (p3_....py) runs over all available volunteers
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
for metric in metrics:
    dataMat[metric] = {}
    print('Current metric is {}'.format(metric))
    for dim in range(maxdims[metric]+1):
        dataMat[metric][dim] = np.full([lan,lan],1000.0)
        dataMat[metric][dim][np.diag_indices(lan)] = 0

    locBeg = basedir[metric] + '{}AllDist_HN/Begun/'.format(metric)
    makedirs(locBeg ,exist_ok=True)
    begun = p_utils.getSecondList(locBeg)

    locFin = basedir[metric] + '{}AllDist_HN/Finished/'.format(metric)
    makedirs(locFin ,exist_ok=True)
    finished = p_utils.getSecondList(locFin)

# Load distances
for metric in metrics:

    try:

        print('Pulling existing save data for metric {}'.format(metric))
        locEnd = basedir[metric] + 'results/{}AllDist_HN/Saved/'.format(metric.lower())
        file = locEnd + 'aggMatrix.pkl'
        with open(file, 'rb') as loadfile:
            dataMat[metric] = pickle.load(loadfile)
        print('savedata is len {}'.format(len(dataMat[metric])))

    except FileNotFound:
        for fin in tqdm(finished,desc='Load data'):
            fname = locFin + fin + '.npy'
            tempD = np.load(fname)
            fname = locBeg + fin + '.npy'
            otherInds = np.load(fname)
            if len(tempD):
                for dim in range(maxdim+1):
                    try:
                        dataMat[metric][dim][allNames_dict[fin],otherInds] = tempD[:,dim].ravel()
                        dataMat[metric][dim][otherInds,allNames_dict[fin]] = tempD[:,dim].ravel()
                    except ValueError:
                        print([dim, ': ', fname, tempD.shape, otherInds.shape])

            
def updateMapping(metric, intiger, nVols = 100, dropSome=False):
            
    locFin = basedir[metric] + 'results/{}AllDist_HN/Finished/'.format(metric)    
    finished = p_utils.getSecondList(locFin)              
        
    allFin = np.array(list(allNames_dict[fin] for fin in finished))
    T = volInds[all_times_vols[0]][-1]+1
    
    allDembs = {}
    
    # trim finished
    someFin = []
    if nVols > -1:
        initDrop = min(nVols*len(all_times_vols)*3,len(allFin))
        print('initially trimming allFin from len {} to len {}'.format(len(allFin),initDrop))
        allFin = np.random.choice(allFin,initDrop)
        print('Trimming allFin from len {}'.format(len(allFin)))
        for vi, voln in tqdm(enumerate(all_times_vols), desc='Trimming group embedding inputs.'):  
            if dropSome & any([voln.find(dropVol)==3 for dropVol in ['14','10','01']]):
                continue
            bounds = (vi*T, (vi+1)*T-1)
            vec = allFin[(allFin>=bounds[0]) * (allFin<=bounds[1])]
            while len(vec)>nVols:
                vecd = np.diff(vec)
                vecs = np.sum(np.concatenate([vecd[:-1][:,None],vecd[1:][:,None]],axis=1),axis=1)
                pop = np.argmin(vecs)
                vec = np.delete(vec,pop+1)
            someFin.extend(vec)
        allFin = someFin        
        print('allFin trimmed to len {}'.format(len(allFin)))
    
    print('Make group embedding')
    embedG = {}        
    #for dim in range(maxdims[metric]+1):
    for dim in rundims[metric]:
        
        groupG = dataMat[metric][dim][allFin,:][:,allFin]
        reducer = umap.UMAP(n_neighbors=200, n_components=2, metric=p_utils.donotripit, n_epochs=1000, learning_rate=5.0, init='random', min_dist=0.3, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=2.0, repulsion_strength=2.0, negative_sample_rate=5, transform_queue_size=16.0, a=None, b=None, random_state=intiger, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, verbose=True)
        embedG[dim] = reducer.fit(groupG)
        print(np.histogram(embedG[dim].embedding_))

    print(len(embedG))
    for voln in all_times_vols:          
        print('Make {} embedding'.format(voln))
        Dembs = {}
        #for dim in range(maxdims[metric]+1):
        for dim in rundims[metric]:

            volFin = volInds[voln]
            group = dataMat[metric][dim][volFin,:][:,allFin]
            print(group.shape)
            Dembs[dim] = embedG[dim].transform(group)

            print(np.histogram(Dembs[dim]))                

        loc = basedir[metric] + 'results/UMAPxyAll{}/'.format(metric)
        print(metric)
        print(loc)
        
        allDembs[str(voln)] = Dembs
        
        #makedirs(loc ,exist_ok=True)
        #file = (loc + str(voln) + '.pkl')
        #with open(file, 'wb') as sfile:
        #    pickle.dump(Dembs,sfile, pickle.HIGHEST_PROTOCOL)        
        
    return allDembs
        
# do random processing        
        
randomIntigers = np.random.randint(1,1024,size=3)        

for intiger in randomIntigers:
    
    allDembs_metrics = {}
    
    for metric in metrics:
        allDembs_metrics[metric] = updateMapping(metric, intiger, nVols = 100, dropSome=False)
        
    # Get UMAP data
    h3 = {}
    for metric in metrics:    
        # Get underlying data
        #for dim in range(maxdims[metric]+1):
        for dim in rundims[metric]:
            condition_label = '{}_{}'.format(metric,dim)
            h3[condition_label] = []

        volunteers = p_utils.getSecondList(localdir[metric])
        for vi, vol in tqdm(enumerate(volunteers),desc='load embedding for metric {}'.format(metric)):
            temp = allDembs_metrics[metric][str(vol)]
            
            #for dim in range(maxdims[metric]+1):
            for dim in rundims[metric]:
                condition_label = '{}_{}'.format(metric,dim)
                h3[condition_label].append(temp[dim])                                        
                    
    e_ImgOuts = {}
    v_ImgOuts = {}

    points = {}
    F = {}
    xx = {}
    yy = {}
    wtr = {}

    esig = {}
    vsig = {}
    psig = {}

    entropy_outs = {}
    cluster_outs = {}
    significant_outs = {}

    barChart = {}
    barChart_full = {}
    
    barChart_randtot = {}
    barChart_randsub = {}

    vBarOut = {}

    p_routs = {}
    p_distribs = {}

    w_stats = {}

    for condition in conditions:
        metric, dim = [*conditions[condition]]
        condition_label = '{}_{}'.format(*conditions[condition])
        print(condition_label)

        data = np.concatenate([h3[condition_label][nv] for nv in range(Nv)])
        print('data is shape {}'.format(data.shape))
        data_vec = np.arange(data.shape[0])
        xx[condition_label], yy[condition_label], xmin, xmax, ymin, ymax = setFed(data)
        if xmax-xmin > ymax-ymin:
            for nv in range(Nv):
                h3[condition_label][nv] = np.roll(h3[condition_label][nv], 1, axis=1)
            data = np.roll(data, 1, axis=1)                
            xx[condition_label], yy[condition_label], xmin, xmax, ymin, ymax = setFed(data)
        snap_fun = partial(p_utils.snapInd,myGrid=np.linspace(xmin,xmax,num=xx[condition_label].shape[0]))
        xsnap = list(map(snap_fun,data[:,0]))
        snap_fun = partial(p_utils.snapInd,myGrid=np.linspace(ymin,ymax,num=yy[condition_label].shape[1]))
        ysnap = list(map(snap_fun,data[:,1]))
        points[condition_label] = np.ravel_multi_index([xsnap,ysnap],xx[condition_label].shape)
        #points[condition_label] = geo.MultiPoint(np.array([xsnap,ysnap]).T)                        

        F[condition_label], kFactor = getFed(data, xx=xx[condition_label], yy=yy[condition_label])

        wtr[condition_label] = watershed(1-F[condition_label], mask=F[condition_label]>0.000001)
        uwtr = np.unique(wtr[condition_label])

        entropy = np.full(len(uwtr)+1,1,dtype='float')

        esig[condition_label] = sig/(len(uwtr)*len(types_ce))
        vsig[condition_label] = sig/(1*len(types_cv))

        entropy_outs[condition_label] = np.full(len(CV),1,dtype='float')
        cluster_outs[condition_label] = np.full(len(CV),0,dtype='int')        
        significant_outs[condition_label] = np.full(len(CV),False)        

        e_routs = pd.DataFrame(columns=types_ce,dtype=float)
        e_ImgOuts[condition_label] = np.zeros(wtr[condition_label].shape)

        p_routs[condition_label] = {}    
        for e in types_ce:
            p_routs[condition_label][e] = pd.DataFrame(columns=list(performance.keys()),dtype=float)        

        numEinW = np.zeros(mxe1)

        v_routs = pd.DataFrame(columns=types_cv,dtype=float)
        v_ImgOuts[condition_label] = -np.ones(wtr[condition_label].shape)

        barChart[condition_label] = {}
        barChart_full[condition_label] = {}
        barChart_temp = {}
        
        barChart_randtot[condition_label] = {}
        barChart_randsub[condition_label] = {}
        barChart_randtmp = {}
        r_routs = pd.DataFrame(columns=types_ce,dtype=float)

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
        e_sum_w = {}    
        for e in types_ce:
            barChart[condition_label][e] = 0            
            barChart_full[condition_label][e] = 0                        
            barChart_randtot[condition_label][e] = 0
            barChart_randsub[condition_label][e] = 0
            lineDraw[e] = set()
            lineDraw_full[e] = set()
            proportions_ref[e] = np.sum(CE==e)
            e_sum_w[e] = 0
        e_all_w = {}
        p_distribs[condition_label] = {}

        w_stats[condition_label] = {}

        vBarOut[condition_label] = np.zeros(len(types_cv))

        print('Set has {} clusters'.format(len(uwtr)))
        for w in uwtr:
            #print('Doing wtr region # ' + str(w))
            #rows, cols = np.nonzero(wtr[condition_label]==w)
            #pts = [[xx[condition_label][rows[i],0], yy[condition_label][0,cols[i]]] for i in range(len(rows)) ]
            #obj = geo.MultiPoint(pts)
            #wpolys = obj.convex_hull
            #wouts = [i for i,k in enumerate(points[condition_label]) if k.intersects(wpolys)]        
            indices_set  = set(np.nonzero((wtr[condition_label]==w).ravel())[0])
            wouts = [i for i,k in enumerate(points[condition_label]) if k in indices_set ]                        

            cluster_outs[condition_label][wouts] = w                

            if not wouts:
                #print('no points under wtr region # ' + str(w))
                e_routs.loc[w,:] = np.nan 
                v_routs.loc[w,:] = np.nan 
                r_routs.loc[w,:] = np.nan 
                entropy[w] = 1#np.nan                
                continue
            else:
                w_stats[condition_label][w] = {}
                #print('Site {} of watershed holds {} points'.format(w,len(wouts)))
                #pass

            w_stats[condition_label][w]['points_all'] = np.array(wouts)
                                        
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

                w_stats[condition_label][w][e] = {'points':wpts, 'npoints':len(wpts), 
                                                  'sig_bool':p_value<esig[condition_label],
                                                  'pvalue':p_value}
                numEinW[e] = obs      
                
                randCE = np.random.permutation(CE)
                epts = np.array([randCE[pt]==e for pt in wouts])
                obs = np.sum(epts)
                zstat = -(rmea-obs)/rstd                
                p_value = st.norm.sf(zstat)
                r_routs.loc[w,e] = p_value            
                barChart_randtmp[e] = obs

            sum_w = np.sum(numEinW)
            normalizer = np.log2(np.sum(numEinW>0))
            entropy[w] = -np.sum([obs/sum_w * np.log2(obs/sum_w) / normalizer for obs in numEinW if obs != 0])
            entropy_outs[condition_label][wouts] = entropy[w]

            rands = [np.random.choice(CV, size=len(wouts), replace=False) for _ in range(300)]
            for v in types_cv:
                dist = np.array([np.sum(rr==v) for rr in rands]).ravel()
                obs = np.sum([CV[ind]==v for ind in wouts])
                rmea = np.mean(dist)
                rstd = np.std(dist)
                # For volunteers, test if not less-than mean
                zstat = (rmea-obs)/rstd 
                p_value = st.norm.sf(zstat)
                v_routs.loc[w,v] = p_value                    

            e_all = e_routs.columns.values[np.array(e_routs.loc[w,:].values < esig[condition_label]).astype('bool')]
            e_all_w[w] = e_all
            if any(e_all):                                
                e_best = e_routs.columns.values[np.argmin(e_routs.loc[w,:].values)]
                for ei in list(e_all):
                    boxPlot[w,ei] = boxPlot_all[w,ei]
                    barChart_full[condition_label][ei] += barChart_temp[ei]                    
                    barChart_randsub[condition_label][ei] += barChart_randtmp[ei]
                    proportions[w,ei] = proportions_all[w,ei]
                    lineDraw_full[ei].update(list(lineDraw_temp[ei]))               
                    sigInds = (cluster_outs[condition_label]==w) & (CE==ei)                    
                    significant_outs[condition_label][sigInds] = True
                    e_sum_w[ei] += 1

                if np.sum(e_routs.loc[w,:].values < esig[condition_label])>1:
                    e = mxe1
                else:
                    e = e_best
                    barChart[condition_label][e] += barChart_temp[e]                    
                    lineDraw[e].update(list(lineDraw_temp[e]))                
            else:
                e_best = e = mne

            e_ImgOuts[condition_label][wtr[condition_label]==w] = e
            
            # Testing for the randomized labels, a null distribution over outside loop iterations
            r_all = r_routs.columns.values[np.array(r_routs.loc[w,:].values < esig[condition_label]).astype('bool')]
            if any(r_all):                                
                for ei in list(r_all):
                    barChart_randtot[condition_label][ei] += barChart_randtmp[ei]

            # Testing how many volunteers appear less than expected inside cluster
            vsum = np.sum(v_routs.loc[w,:].values < vsig[condition_label])
            v = vsum#mxv1            
            v_ImgOuts[condition_label][wtr[condition_label]==w] = v
            vBarOut[condition_label][v] += len(wouts)
            w_stats[condition_label][w]['vols'] = v

        print('total points in vBarOut is {}'.format(np.sum(vBarOut[condition_label])))

        psig[condition_label] = {}
        for e in types_ce[2:]:
            if e_sum_w[e]:
                psig[condition_label][e] = sig/e_sum_w[e]
            else:
                psig[condition_label][e] = 0

        # check performance
        for w in uwtr:
            indices_set  = set(np.nonzero((wtr[condition_label]==w).ravel())[0])
            wouts = [i for i,k in enumerate(points[condition_label]) if k in indices_set ]        
            p_distribs[condition_label][w] = {}

            cluster_outs[condition_label][wouts] = w                

            if not wouts:            
                continue
            else:
                pass

            # With respect to the number of points for a given experiment lying within a 
            # cluster significant for that experiment, test the hypothesis that a performance metric
            # is less than or greater than the mean performance metric for that experiment
            for e in types_ce[2:]:
                lenw = np.sum(CE[wouts]==e)
                if lenw > 0 and any(e_all_w[w]==e):
                    ThisDataVec = data_vec[CE==e]
                    rands = [np.random.choice(ThisDataVec, size=lenw, replace=False) for _ in range(300)]            
                    p_distribs[condition_label][w][e] = {}
                    for perf in performance:
                        distrabution = np.array([np.mean(performance[perf][rrs]) for rrs in rands]).ravel()
                        rmea = np.mean(distrabution)
                        rstd = np.std(distrabution)
                        epts = np.array([performance[perf][pt] for pt in wouts if CE[pt] == e])
                        obs = np.mean(epts)
                        zstat = -(rmea-obs)/rstd
                        p_value = st.norm.sf(np.abs(zstat)) * 2 
                        p_distribs[condition_label][w][e][perf] = {'rmea':rmea, 'rstd':rstd, 'obs':obs, 'p':p_value}
                        if p_value < psig[condition_label][e]:
                            p_routs[condition_label][e].loc[w,perf] = ((-1) ** (zstat<0)) # p_value -> sig less than or greater than

        for e in types_ce[2:]:    
            p_routs[condition_label][e].fillna(0, inplace=True)                

                        
    # Do the plots

    fig = plt.figure(num = 22,figsize=[2*7,2*4.5])

    maxVBarOut = np.max(list([vBarOut[clbl].astype(int)] for clbl in vBarOut))

    for i, condition in enumerate(conditions):
        condition_label = '{}_{}'.format(*conditions[condition])

        ax = plt.subplot(6, 6, 1+i) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[condition_label][nv][:,0])
            Y.append(h3[condition_label][nv][:,1])
            c.append(cv[nv])
            s.append((ct[nv]!=-11).astype('int')*1)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.2,cmap='viridis',vmin=-1,vmax=mxv1)
        #cnt = ax.contour(yy[condition_label],-xx[condition_label], F[condition_label], alpha=0.3)
        ax.contour(yy[condition_label],-xx[condition_label],F[condition_label], alpha=0.4, linestyles='-')
        ax.set_aspect('equal','box')            
        if i==5:
            cbar = plt.colorbar(ax=[ax])
            cbar.ax.set_ylabel('volunteer #')
        if i==0:
            ax.set_ylabel('A')
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(condition_label)


        ax = plt.subplot(6, 6, 7+i) 
        img = ax.imshow(Nv-v_ImgOuts[condition_label], cmap='viridis',alpha=0.5,vmin=-1,vmax=mxv1)      
        #cbar.set_ticks( [-1] + list(types_cv) + [mxv1] )
        #cbar.ax.set_xticklabels(['None'] + list('Vol_{}'.format(v) for v in types_cv) + ['Multi'], rotation=90)
        ax.contour(F[condition_label], alpha=0.5)
        #plt.suptitle( (savename + '\n' + 'Dim {} | nVol {}'.format( dim, Nv) + '\n' + 'ExpSig {:.3e} | VolSig {:.3e} | bandwidth {:.2f}'.format( esig[condition_label],vsig[condition_label],kFactor[condition_label]) ) )        
        if i==5:
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.set_ylabel('# volunteers\n represented')
            #ax.set_ylabel('Volnt. clusters')
        if i==0:
            ax.set_ylabel('B')
        ax.set_xticks([],[])
        ax.set_yticks([],[])


        ax = plt.subplot(6, 6, 13+i)     
        labels = types_cv
        x = np.arange(len(labels))+1  # the label locations
        #width = 0.35  # the width of the bars
        v_frac = np.cumsum(vBarOut[condition_label][::-1].astype(int))
        v_frac2 = v_frac/np.max(v_frac)*100
        rects1 = ax.bar(x , v_frac2)#, width, color='r')
        v_num = np.sum(np.multiply(vBarOut[condition_label][::-1],np.arange(1,19))).astype('int')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylim(bottom=0, top=100)
        ax.set_xticks([0,5,10,15])
        ax.grid(b=True,which='both',axis='y')
        if i == 5:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel('cumsum %y points\nwith #x adjacent\nvolunteers.')        
        else:        
            ax.yaxis.tick_right()
            ax.set_yticklabels([])    
        if i==0:
            ax.yaxis.set_label_position("left")
            ax.set_ylabel('C')
            ax.text(0,80,'sum = ' + str(v_num))
        else:
            ax.text(0,80,str(v_num))


        ax = plt.subplot(6, 6, 19+i) 
        X , Y, c, s = [], [], [], []
        for nv in range(Nv):        
            X.append(h3[condition_label][nv][:,0])
            Y.append(h3[condition_label][nv][:,1])
            c.append(ce[nv].ravel())
            s.append((ce[nv]!=-11).astype('int')*1)
        ppl = plt.scatter(np.concatenate(Y).ravel(),-np.concatenate(X).ravel(),
                    c=np.concatenate(c).ravel(),s=np.concatenate(s).ravel(),
                    marker='.',alpha=0.2,cmap='nipy_spectral',vmin=0,vmax=mxe1)
        #cnt = ax.contour(yy[condition_label], -xx[condition_label], F[condition_label], alpha=0.3)
        ax.contour(yy[condition_label],-xx[condition_label],F[condition_label], alpha=0.4, linestyles='-')
        if i==5:        
            axs = []
            axs.append(ax)
        if i==0:
            ax.set_ylabel('D')
        ax.set_aspect('equal','box')     
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)


        ax = plt.subplot(6, 6, 25+i) 
        img = ax.imshow(e_ImgOuts[condition_label], cmap='nipy_spectral',alpha=0.5,vmin=0,vmax=mxe1)
        ax.contour(F[condition_label], alpha=0.5)        
        if i==5:
            axs.append(ax)
            cbar = plt.colorbar(img, ax=axs)
            cbar.set_ticks([0] + list(types_ce) + [mxe1])
            cbar.ax.set_yticklabels( goodStates + ['Multi'])
            cbar.ax.set_ylabel('stimulus type')
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        if i==0:
            ax.set_ylabel('E')


        ax = plt.subplot(6, 6, 31+i)     
        labels = goodStates[1:]
        labels.append('Totals')
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        e_frac_full = list(barChart_full[condition_label][e]/np.sum(CE==e)*100 for e in types_ce)
        tot_full = np.sum([barChart_full[condition_label][e] for e in barChart_full[condition_label]])/len(CE)*100
        e_frac_full.append(tot_full)
        rects1 = ax.bar(x , e_frac_full, width, color='r')
        
        e_frac = list(barChart[condition_label][e]/np.sum(CE==e)*100 for e in types_ce)
        tot = np.sum([barChart[condition_label][e] for e in barChart[condition_label]])/len(CE)*100
        e_frac.append(tot)
        rects2 = ax.bar(x , e_frac, width, color='b')        
        
        r_frac_sub = list(barChart_randsub[condition_label][e]/np.sum(CE==e)*100 for e in types_ce)
        tot = np.sum([barChart_randsub[condition_label][e] for e in barChart_randsub[condition_label]])/len(CE)*100
        r_frac_sub.append(tot)        
        rects3 = ax.bar(x , r_frac_sub, width, color='y')
        
        r_frac_tot = list(barChart_randtot[condition_label][e]/np.sum(CE==e)*100 for e in types_ce)
        tot = np.sum([barChart_randtot[condition_label][e] for e in barChart_randtot[condition_label]])/len(CE)*100
        r_frac_tot.append(tot)
        rects4 = ax.bar(x , r_frac_tot, width, color='g')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        if i == 5:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel('% points within\nclusters significant\n to each stimuli')        
        else:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_yticks([],[])
        if i==0:
            ax.yaxis.set_label_position("left")
            ax.set_ylabel('F')
        #ax.set_title(( savename + 'Dim {} | nVol {}'.format( dim,Nv) ))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation = 70)     
        ax.set_ylim(bottom=0, top=100)
        for rect in rects1:
            height = rect.get_height()
            ax.annotate('{:.0f}'.format(height),
                            xy=(rect.get_x() + rect.get_width(), min(80,height+20)),
                            xytext=(-2, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', c='r')
        for rect in rects2:
            height = rect.get_height()
            ax.annotate('{:.0f}'.format(height),
                            xy=(rect.get_x() + rect.get_width(), height),
                            xytext=(-1, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', c='b')

        #autolabel(rects1)
        #autolabel(rects2)

    #fig.tight_layout()    
    name = './figures/Figure2_randInt{}.png'.format(intiger)
    plt.savefig(name)
    plt.close()                
    
    fig = plt.figure(num = 32,figsize=[5.5*2,7*2])

    colors = ['b','r']
    markers = ['.','.','.','2','3','4']
    hatches = [ ['.', '.', '.', '--', '||', '//'] , ['.', '.', '.', '...', 'oo', '*'] ]

    axs = []

    for i, condition in enumerate(conditions):
        condition_label = '{}_{}'.format(*conditions[condition])
        metric, dim = [*conditions[condition]]

        for ei, e in enumerate(types_ce[2:], start=0):            
            for pi, perf in enumerate(performance,start=0):        
                ax = plt.subplot(9, 8, 1+i + 8*(pi+3*ei) )
                temp = wtr[condition_label]*0

                for w in p_routs[condition_label][e].index:
                    val = int(p_routs[condition_label][e].loc[w,[perf]][0])
                    temp[wtr[condition_label]==w] = val
                img = ax.imshow(temp, cmap='RdYlGn', vmin=-1, vmax=1)
                ax.contour(F[condition_label],alpha=0.4)              
                ax.set_xticks([],[])
                ax.set_yticks([],[])
                if i==0:
                    ax.set_ylabel(goodStates[e] + '\n' + perf )
                if pi==0 and ei==0:
                    ax.set_title(condition_label)                                

    colors = ['r','g']
    markers = ['$e$','$a$','$d$','$0$','$1$','$2$']

    vals = {}

    for ei, e in enumerate(types_ce[2:], start=0):            
        vals[e] = {}    
        for pi, perf in enumerate(performance):
            vals[e][perf] = {}
            for i, condition in enumerate(conditions):
                condition_label = '{}_{}'.format(*conditions[condition])            
                vals[e][perf][condition_label] = {'v-1':[], 'v0':[], 'v1':[], 
                                                  'n-1':0, 'n0':0, 'n1':0,
                                                 'p-1':[], 'p0':[], 'p1':[],
                                                 'a-1':[], 'a0':[], 'a1':[]}
                
    for i, condition in enumerate(conditions):
        condition_label = '{}_{}'.format(*conditions[condition])
        for ei, e in enumerate(types_ce[2:], start=0):            
            for w in p_routs[condition_label][e].index:
                for pi, perf in enumerate(performance):
                    perf_type = p_routs[condition_label][e].loc[w,perf]
                    if perf_type == 0:
                        vals[e][perf][condition_label]['v0'].append(Nv-w_stats[condition_label][w]['vols'])        
                        vals[e][perf][condition_label]['n0'] += w_stats[condition_label][w][e]['npoints']
                        vals[e][perf][condition_label]['p0'].append(w_stats[condition_label][w][e]['points'])                     
                        vals[e][perf][condition_label]['a0'].append(w_stats[condition_label][w]['points_all'])
                    elif perf_type == -1:
                        vals[e][perf][condition_label]['v-1'].append(Nv-w_stats[condition_label][w]['vols'])        
                        vals[e][perf][condition_label]['n-1'] += w_stats[condition_label][w][e]['npoints']
                        vals[e][perf][condition_label]['p-1'].append(w_stats[condition_label][w][e]['points'])
                        vals[e][perf][condition_label]['a-1'].append(w_stats[condition_label][w]['points_all'])
                    elif perf_type == 1:
                        vals[e][perf][condition_label]['v1'].append(Nv-w_stats[condition_label][w]['vols'])        
                        vals[e][perf][condition_label]['n1'] += w_stats[condition_label][w][e]['npoints']
                        vals[e][perf][condition_label]['p1'].append(w_stats[condition_label][w][e]['points'])        
                        vals[e][perf][condition_label]['a1'].append(w_stats[condition_label][w]['points_all'])
                        
    xlims = [0,19]
    ylims = [-1,0]                    

    axs = []    
    for pi, perf in enumerate(performance):
        for ei, e in enumerate(types_ce[2:], start=0):            
            ax = plt.subplot(9,8, 7 + 8*(pi+3*ei))  
            if pi==0 and ei==0:
                ax.set_title('Counts A')
            if pi==2 and ei==2:
                ax.set_xlabel('mean volunteers')
            axs.append(ax)
            for i, condition in enumerate(conditions):
                condition_label = '{}_{}'.format(*conditions[condition])
                for k in [-1,1]:
                    yy = vals[e][perf][condition_label]['n' + str(k)]
                    if yy>0:
                        xx = np.mean(vals[e][perf][condition_label]['v' + str(k)])
                        ax.plot(xx,yy,c=colors[int(k==1)],marker=markers[i],alpha=0.5)   
                        #print([e,perf,condition_label, yy])
            temp = ax.get_ylim()
            ylims[1] = max(ylims[1],temp[1])

    ylims[1] *= 1.05       
    for ax in axs:
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        ax.set_yticks([0,1000])
        ax.set_yticklabels([0,'1k'])

    xlims = [-1,0]
    ylims = [-1,0]                    

    pars = []      
    for pi, perf in enumerate(performance):
        for ei, e in enumerate(types_ce[2:], start=0):            
            ax = plt.subplot(9, 8, 8 + 8*(pi+3*ei))        
            if pi==0 and ei==0:
                ax.set_title('Counts B')
            if pi==2 and ei==2:
                ax.set_xlabel('performance')
            counts, bins, img = ax.hist(performance[perf][CE==e],bins=30,alpha=0.2,color='y')
            perf_mean = np.mean(performance[perf][CE==e])
            ax.plot([perf_mean]*2,[0,max(counts)],c='k',alpha=0.7,linestyle=':')
            #par = ax.twinx()
            #pars.append(par)
            for i, condition in enumerate(conditions):
                condition_label = '{}_{}'.format(*conditions[condition])
                for k in [-1,1]:
                    yy = vals[e][perf][condition_label]['n' + str(k)]
                    if yy>0:
                        inds = vals[e][perf][condition_label]['p' + str(k)]
                        inds = np.concatenate(inds)                    
                        xx = np.mean(performance[perf][inds])
                        ax.plot(xx,yy,c=colors[int(k==1)],marker=markers[i],alpha=0.5)   
                        #print([e,perf,condition_label, yy])
            ax.plot([perf_mean]*2,[0,ax.get_ylim()[1]],c='k',alpha=0.7,linestyle=':')
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            #temp = par.get_xlim()
            #xlims[0] = min(xlims[0],temp[0])
            #xlims[1] = max(xlims[1],temp[1])
            #temp = par.get_ylim()        
            #ylims[0] = min(ylims[0],temp[0])
            #ylims[1] = max(ylims[1],temp[1])

    #ylims[1] *= 1.05       
    #xlims[1] *= 1.05        
    #for ax in pars:
        #ax.set_ylim(ylims)
        #ax.set_xlim(xlims)            

    #######    

    fig.tight_layout()
    name = './figures/Figure3_randInt{}.png'.format(intiger)
    plt.savefig(name)
    plt.close()                
        
    name = './figures/resultsSaves_randInt{}.pkl'.format(intiger)
    with open(name, 'wb') as file:
        pickle.dump({'vBarOut': vBarOut, 'barChart_full': barChart_full, 'vals':vals,
                     'barChart_randtot': barChart_randtot, 'barChart_randsub': barChart_randsub,
                    'barChart': barChart}, file)
        
        
