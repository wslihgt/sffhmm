#!/usr/bin/python

import numpy as np
import os.path
import time
import ARPIMM
import sys

import scipy.linalg as spla

## import scikits.audiolab as al
import scipy.io.wavfile as wav

from tracking import viterbiTrackingArray

sys.path.append('../tools')
from imageTools import imageM, subplotMatrix, plotRegions
from imageTools import plotRegionsX, plotRegionsY

import readSVLfiles
from matplotlib.mlab import find as mlabFind

from SFSNMF_functions import *

# Running the tests of above functions:

displayEvolution = True

import matplotlib.pyplot as plt
from imageMatlab import imageM

## plt.rc('text', usetex=True)
plt.rc('image',cmap='jet')
plt.ion()

windowSizeInSamples = 2048
hopsize = 256
NFT = 2048
niter = 50

# TODO: also process these as options:
minF0 = 80
maxF0 = 500
Fs = 44100.0
maxFinFT = 8000.0
F = np.ceil(maxFinFT * NFT / np.double(Fs)) 

stepNotes = 16 # this is the number of F0s within one semitone
K = 10 # number of spectral shapes for the filter part
R = 0 # number of spectral shapes for the accompaniment
P = 30 # number of elements in dictionary of smooth filters

# number of chirped spectral shapes between each F0
# this feature should be further studied before
# we find a good way of doing that.
chirpPerF0 = 1

# Create the harmonic combs, for each F0 between minF0 and maxF0:

F0Table, WF0 = \
         generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, 
                              stepNotes=stepNotes, 
                              lengthWindow=windowSizeInSamples, Ot=0.9, 
                              perF0=chirpPerF0, 
                              depthChirpInSemiTone=.15,
                              loadWF0=True,
                              analysisWindow='hanning')

WF0 = WF0[0:F, :] # ensure same size as SX 
NF0 = F0Table.size # number of harmonic combs
# Normalization: 
WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))

withExtraUnvoiced = True
if withExtraUnvoiced:
    WF0 = np.concatenate((WF0, np.ones([F,1])), axis=1)
    NF0 = NF0 + 1



bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = genARbasis(
    F, NFT, Fs, maxF0=2*maxF0, formantsRange=formantsRange)
numberOfFormants = freqRanges.shape[0]
numberOfAmpPerFormantFreq = bwRange.size

WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
Fwgamma, Nwgamma = WGAMMA.shape

nbElPerF = Nwgamma/numberOfFormants

## added 20110401 for the picture in icassp poster!
args = ['/users/jeanlouis/work/BDD/formants/presentingSNNDSPSFFT.wav']

fs, data = wav.read(args[0])
data = data / np.double(np.abs(data).max())

X= stft(data, fs=fs, hopsize=hopsize,
               window=sinebell(windowSizeInSamples), nfft=NFT)
SX = np.abs(X[0]) ** 2
SX = SX[:F,:]

if False:
    W = [WF0.copy()]
    nElPerFor = Nwgamma/numberOfFormants
    for p in range(numberOfFormants):
        W.append(np.hstack((WGAMMA[:,(p*nElPerFor):((p+1)*nElPerFor)],
                            np.atleast_2d(np.ones(F)).T)))
        # adding a vector of ones, this is to "deactivate"
        # the corresponding W...
    
    F,N = SX.shape
    P = len(W)
    K = np.zeros(P) # each matrix size
    for p in range(P):
        if W[p].shape[0] == F: K[p] = W[p].shape[1]
        else: raise ValueError('Size of W[%d] not same as input SX' %(p))
    
    ##S = [np.dot(W[0], HF00)]
    ##for p in range(1,P):
    ##    S.append(np.dot(W[p],HPHI0))
    
    ##SX = np.prod(S, axis=0)
    
    ##SX = np.atleast_2d(SX).T
    ##SX = hstack([SX,SX,SX,SX,SX,SX,SX,SX,SX,SX,
    ##             SX,SX,SX,SX,SX,SX,SX,SX,SX,SX,SX,SX,])
    ##SX = SX[:,np.zeros(10, dtype=np.int32)]
    
    H, recoError1 = ARPIMM.ARPIMM(SX, W, nbIterations=50)
    if False:
        ## display reweighing scheme:
        plt.rc('text', usetex=True)
        fontsize = 36
        frameNo = 100
        p = 0
        sigma = 1000
        eps = 1e-50
        muH = np.dot(np.arange(K[p]-1) * \
                     (np.arange(K[p]-1, 0, -1))**2, \
                     H[p][:-1,:]) / \
                     np.dot((np.arange(K[p]-1, 0, -1))**2,
                            np.maximum(H[p][:-1,:], eps))
        Hp = np.exp(- 0.5 * ((np.outer(np.arange(K[p]),
                                       np.ones(N)) - \
                              np.outer(np.ones(K[p]),
                                       muH)) ** 2) / \
                    np.outer(np.ones(K[p]), sigma))
        plt.figure()
        plt.plot(H[p][:,frameNo], label=u'$\mathbf{h}_n^p$')
        plt.plot([muH[frameNo], muH[frameNo]],[0, H[p][:,frameNo].max()],
                 color='r', label=u'$\mu_{pn}$')
        plt.plot(Hp[:,frameNo]*H[p][:,frameNo].max(), color='g', ls='--', 
                 label=u'Weight')
        plt.plot(Hp[:,frameNo]*H[p][:,frameNo], color='g',
                 label=u'Reweighed $\mathbf{h}_n^p$')
        plt.legend(prop={'size': fontsize})
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('index $k$', fontsize=fontsize)
    if False:
        ## display reweighing smoothing scheme:
        plt.rc('text', usetex=True)
        fontsize = 36
        frameNo = 100
        p = 0
        sigma = 1000
        eps = 1e-50
        muH = np.dot(np.arange(K[p]-1) * 
                     (np.arange(K[p]-1, 0, -1))**2, 
                     H[p][:-1,:]) / \
                     np.dot((np.arange(K[p]-1, 0, -1))**2,
                            np.maximum(H[p][:-1,:], eps))
        Hp = np.exp(- 0.5 * ((np.outer(np.arange(K[p]),
                                       np.ones(N)) - \
                              np.outer(np.ones(K[p]),
                                       muH)) ** 2) / \
                    np.outer(np.ones(K[p]), sigma))
        
        plt.figure()
        imageM(db(H[p]))
        plt.plot(muH, color='k', label=u'$\{\mu_{pn}\}_n$')
        plt.legend(prop={'size': fontsize})
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('frame $n$', fontsize=fontsize)
        plt.ylabel('index $k$', fontsize=fontsize)
        plt.clim([db(H[p]).max()-50, db(H[p]).max()])
        plt.axis('tight')
        subplots_adjust(left=.09, bottom=.20, right=.96, top=.96,)
        cb = plt.colorbar(fraction=0.05)
        
    H, recoError2 = ARPIMM.SparARPIMM(SX, W, poleFrq=None, H0=H, \
                                      stepNotes=None, nbIterations=100,
                                      scopeMedian=6)
    
    parBef, fraBef, durBef, labBef, valBef = readSVLfiles.extractSvlAnnotRegionFile(args[0][:-3]+'svl')
    endBef = fraBef+durBef
    
    eps = 10**-50
    Shat = [np.dot(W[0], H[0])]
    mu = [np.dot(np.arange(NF0-1) * \
                 (np.arange(NF0-1, 0, -1))**2, \
                 H[0][:-1,:]) / \
          np.dot((np.arange(NF0-1, 0, -1))**2,
                 np.maximum(H[0][:-1,:], eps))]
    
    ##plt.figure()
    ##plt.subplot(P,1,1)
    ##imageM(db(Shat[0]))
    for p in range(1,P):
        mu.append(np.dot(np.arange(nbElPerF) * \
                         (np.arange(nbElPerF, 0, -1))**2,
                         H[p][:-1,:]) / \
                  np.dot((np.arange(nbElPerF, 0, -1))**2,
                         np.maximum(H[p][:-1,:], eps)))
        ##plt.subplot(P,1,p+1)
        Shat.append(np.dot(W[p],H[p]))
        ##imageM(db(Shat[p]))
        
    F0seq = F0Table[np.int32(mu[0])]
    
    plt.figure(2, figsize=(20, 1))
    linewidth = 1
    plt.clf()
    ##imageM(db(np.prod(Shat, axis=0)))
    imageM(db(SX))
    Fformant = []
    for p in range(1,P):
        Fformant.append(poleFrq[np.int32((p-1)*nbElPerF+mu[p])])
        plot(Fformant[p-1]/np.double(fs)*NFT, lw=linewidth)
    
    plt.axis('tight')
    
    imageTools.plotRegionsX(fraBef*fs/np.double(hopsize), \
                            endBef*fs/np.double(hopsize), \
                            vmin=4/5.*F, vmax=4/5.*F, labels=labBef)
    
    ytickslab = np.array([2000, 4000, 6000])
    ytickspos = ytickslab / np.double(Fs) * NFT
    plt.yticks(ytickspos, ytickslab)
    
    xtickspos, xtickslab = plt.xticks()
    xtickspos = xtickspos[1:-1]
    xtickslab = xtickspos * hopsize / np.double(Fs)
    xtickslab = (np.int32(xtickslab*100))/100.
    plt.xticks(xtickspos, xtickslab)
    
    plt.subplots_adjust(bottom=0.2, top=0.99, right=0.999, left=0.03)
    
    SXhat = np.prod(Shat, axis=0)

if False:
    # generate synthetic data
    N = 20
    mu0 = np.random.rand(N) * NF0
    vectorNF0 = np.outer(np.arange(NF0), np.ones(N))
    HF00 = ARPIMM.normalDistributionMat(vectorNF0, mu0, .5*np.ones(N))
    SX = np.dot(WF0, HF00)
    
    mu, recoError = ARPIMM.singleMatGauss(SX, WF0, mu0=None, \
                                          nbIterations=200,
                                          dispEvol=True, verbose=True)

# pseudo inversing WF0 does not give any result...
# pseudoInvWF0 = np.dot(WF0.T, spla.inv(np.dot(WF0, WF0.T))) 
##pseudoInvWF0 = np.dot(spla.inv(np.dot(WF0.T, WF0)),WF0.T) 

if False : # get the data from hillenbrand
    from loadHillenbrand import *
    
    filenb = 46 # for bigdat
    filenbt = mpl.mlab.find(timedatnames==bigdatnames[filenb])[0]
    
    fs, data = loadHill(bigdatnames[filenb])
    Fs=fs
    
    windowSizeInSamples = nextpow2(np.ceil(fs * 0.023))
    hopsize = windowSizeInSamples/8
    NFT = windowSizeInSamples
    maxFinFT = 8000
    F = np.ceil(maxFinFT * NFT / np.double(fs))

    timeInFrame = 0.001 * timedat[filenbt] / np.double(hopsize/np.double(fs))
    timeStampsInFrame = timeInFrame[0] + \
                        (timeInFrame[1]-timeInFrame[0]) * \
                        np.arange(1,9) * 0.1# 10%... 80%
    
    X = stft(data, fs=fs, hopsize=hopsize,
             window=sinebell(windowSizeInSamples), nfft=NFT)
    
    SX = np.abs(X[0]) ** 2
    SX = SX[:F,:]
    
    # draw figure plus formant annotated
    FHzmax = 4000.0
    nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
    ytickslab = np.array([1000, 2000, 3000, 4000])
    ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
    fontsize = 16
    figheight=4.5
    figwidth=9.0
    
    plt.figure(figsize=[figwidth, figheight])
    imageM(db(SX))
    plt.hold(True)
    for n in range(3):
        plt.plot(timeStampsInFrame,
                 bigdat[filenb,5+n+3*np.arange(8)]/np.double(fs)*NFT,
                 'ok')
    
    plt.axis('tight')
    
    bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = genARbasis(F, NFT, Fs, maxF0=2*maxF0, formantsRange=formantsRange)
    numberOfFormants = freqRanges.shape[0]
    numberOfAmpPerFormantFreq = bwRange.size
    
    WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
    Fwgamma, Nwgamma = WGAMMA.shape
    
    nbElPerF = Nwgamma/numberOfFormants
    
    F0Table, WF0 = \
             generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, \
                                  stepNotes=stepNotes, \
                                  lengthWindow=windowSizeInSamples, Ot=0.9, \
                                  perF0=chirpPerF0, \
                                  depthChirpInSemiTone=.15,
                                  loadWF0=False,
                                  analysisWindow='hanning')
    
    WF0 = WF0[0:F, :] # ensure same size as SX 
    NF0 = F0Table.size # number of harmonic combs
    # Normalization: 
    WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))
    
    withExtraUnvoiced = True
    if withExtraUnvoiced:
        WF0 = np.concatenate((WF0, np.ones([F,1])), axis=1)
        NF0 = NF0 + 1
    
    W = [WF0.copy()]
    nElPerFor = Nwgamma/numberOfFormants
    for p in range(numberOfFormants):
        W.append(np.hstack((WGAMMA[:,(p*nElPerFor):((p+1)*nElPerFor)],
                            np.atleast_2d(np.ones(F)).T)))
        # adding a vector of ones, this is to "deactivate"
        # the corresponding W...
    
    P = len(W)

    H, recoError1 = ARPIMM.ARPIMM(SX, W, nbIterations=20)
    H, recoError2 = ARPIMM.SparARPIMM(SX, W, poleFrq=None, H0=H, \
                                      stepNotes=None, nbIterations=150)
    eps = 10**-50
    Shat = [np.dot(W[0], H[0])]
    mu = [np.dot(np.arange(NF0-1) * \
                 (np.arange(NF0-1, 0, -1))**2, \
                 H[0][:-1,:]) / \
          np.dot((np.arange(NF0-1, 0, -1))**2,
                 np.maximum(H[0][:-1,:], eps))]
    
    ##plt.figure()
    ##plt.subplot(P,1,1)
    ##imageM(db(Shat[0]))
    for p in range(1,P):
        mu.append(np.dot(np.arange(nbElPerF) * \
                         (np.arange(nbElPerF, 0, -1))**2,
                         H[p][:-1,:]) / \
                  np.dot((np.arange(nbElPerF, 0, -1))**2,
                         np.maximum(H[p][:-1,:], eps)))
        ##plt.subplot(P,1,p+1)
        Shat.append(np.dot(W[p],H[p]))
        ##imageM(db(Shat[p]))
        
    F0seq = F0Table[np.int32(mu[0])]
    
    plt.figure(2)
    plt.clf()
    ## imageM(db(np.prod(Shat, axis=0)))
    imageM(db(SX))
    Fformant = []
    for p in range(1,P):
        Fformant.append(poleFrq[np.int32((p-1)*nbElPerF+mu[p])])
        plot(Fformant[p-1]/np.double(fs)*NFT)
        
    plt.axis('tight')
    
    SXhat = np.prod(Shat, axis=0)

if False:
    # making the pictures:
    FHzmax = 4000.0
    nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
    ytickslab = np.array([1000, 2000, 3000, 4000])
    ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
    fontsize = 16
    figheight=4.5
    figwidth=9.0
    
    nbF0 = 200
    plt.figure(figsize=[figwidth, figheight])
    plt.plot(db(W[0][:nFmax, nbF0]), color='k')
    plt.xticks(ytickspos, ytickslab, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis('tight')
    plt.legend((['F0 = %dHz' %(F0Table[nbF0])]))
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.06)
    
    ##plt.rc('text', usetex=False)
    plt.figure(figsize=[figwidth, figheight])
    lsRange = ('-', '--', ':')
    N1 = 700
    N2 = 705
    legendThing = {}
    for nn in range(N1, N2):
        plt.plot(db(WGAMMA[:nFmax,nn]), lsRange[(nn-N1)%len(lsRange)],
                 color='k')
        plt.xticks(ytickspos, ytickslab, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        legendThing[nn-N1] = u'%1.2f' %(poleAmp[nn])
    
    plt.legend(legendThing.values(), loc='bottom left')
    plt.axis('tight')
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.06)

if False:
    import imageTools
    # pictures for the poster
    FHzmax = 4000.0
    nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
    ytickslab = np.array([1000, 2000, 3000, 4000])
    ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
    fontsize = 16
    figheight=4.0
    figwidth=4.0
    
    plt.figure(figsize=[figwidth, figheight])
    imageTools.imageM(db(W[0][:nFmax]))
    plt.yticks(ytickspos, ytickslab, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.axis('tight')
    plt.title('F0 = %dHz to %dHz' %(F0Table[0], F0Table[-1]))
    plt.subplots_adjust(top=.93, right=0.96, left=.16, bottom=.07)
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/matW0.pdf')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/matW0.eps')
    
    p=3
    plt.figure(figsize=[figwidth, figheight])
    imageTools.imageM(db(W[3][:nFmax]))
    plt.yticks(ytickspos, ytickslab, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.axis('tight')
    plt.title('Fp = %dHz to %dHz' %(freqRanges[p, 0], freqRanges[p, -1]))
    plt.subplots_adjust(top=.93, right=0.94, left=.16, bottom=.07)
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/matWp.eps')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/matWp.pdf')
    
    # drawing a fake vector of activations:
    figheightVec=4.0
    figwidthVec=0.5
    fakeActiveVec = np.zeros([10,1])
    fakeActiveVec[3] = 1
    plt.figure(figsize=[figwidthVec,figheightVec], )
    imageTools.imageM(fakeActiveVec)
    plt.xticks([0], ['n'], fontsize=fontsize)
    plt.yticks([],[])
    plt.subplots_adjust(top=.94, right=0.94, left=.16, bottom=.07)
    
    figheightVec=4.0
    figwidthVec=1.0
    vecSize = W[0].shape[1]
    fakeActiveVec = np.zeros(vecSize)
    fakeActiveVec[100] = 1
    plt.figure(figsize=[figwidthVec,figheightVec], )
    plt.plot(fakeActiveVec, np.arange(vecSize), linewidth=10)
    plt.xticks([0.5], ['n'], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplots_adjust(top=.94, right=0.94, left=.5, bottom=.07)
    plt.axis('tight')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/vecH0.eps')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/vecH0.pdf')
    
    vecSize = W[p].shape[1]
    fakeActiveVec = np.zeros(vecSize)
    fakeActiveVec[200] = 1
    plt.figure(figsize=[figwidthVec,figheightVec], )
    plt.plot(fakeActiveVec, np.arange(vecSize), linewidth=10)
    plt.xticks([0.5], ['n'], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplots_adjust(top=.94, right=0.94, left=.5, bottom=.07)
    plt.axis('tight')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/vecHp.pdf')
    plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                'durrieu2011icassp/poster/vecHp.eps')
    
    # figure for main equation
    n = 255
    figheightVec=4.0
    figwidthVec=1.0
    yValues = np.arange(F)
    for p in range(P):
        plt.figure(figsize=[figwidthVec,figheightVec])
        plt.plot(db(np.dot(W[p],H[p][:,n])), yValues)
        xtickspos, xtickslab = plt.xticks()
        plt.xticks([.5*xtickspos[0]+.5*xtickspos[-1]], ['n'], \
                   fontsize=fontsize)
        ytickspos, ytickslab = plt.yticks()
        plt.yticks([.5*ytickspos[0]+.5*ytickspos[-1]], ['frequency'], \
                   fontsize=fontsize, rotation=270)
        plt.subplots_adjust(top=.94, right=0.94, left=.40, bottom=.07)
        plt.axis('tight')
        #plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
        #            'durrieu2011icassp/poster/vecHp.pdf')
        plt.savefig('/users/jeanlouis/work/svn/PersoDurrieu/writing/'+
                    'durrieu2011icassp/poster/vecHp_%d.eps' % p)
    
    
