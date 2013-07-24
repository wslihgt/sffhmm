#!/usr/bin/python

"""Running an example of SFSNMF for a file from the Hillenbrand dataset

Description
-----------
This script runs the SFSNMF algorithm on an audio file from the Hillenbrand
dataset. The Source/Filter Sparse NMF algorithm is described in

 Durrieu, J.-L. and Thiran, J.-P.
 \"Source/Filter Factorial Hidden Markov Model,
 with Application to Pitch and Formant Tracking\"
 IEEE Transactions on Audio, Speech and Language Processing,
 Submitted Jan. 2013, Accepted July 2013.

Usage
-----

 1) modify the `loadHillenbrand.py` script such that
    the directories fit those of your system (especially where the program
    can find the Hillenbrand directory)
 2) run the script (preferably within an `ipython` console, for interactivity
    and to be able to play with the figures once the computation is done.).
    Some outputs will be displayed on the console output, in particular the
    progress of the computation as well as the reconstruction errors.
    Some figures are also displayed, showing the evolution of the various
    quantities during the estimation.

2013 - Jean-Louis Durrieu (http://www.durrieu.ch/research/)

"""

import numpy as np
import os.path
import time
import ARPIMM

import scipy.linalg as spla

import matplotlib as mpl
import matplotlib.pyplot as plt

from tracking import viterbiTrackingArray

from SFSNMF_functions import *

displayEvolution = True

import matplotlib.pyplot as plt

## plt.rc('text', usetex=True)
plt.rc('image',cmap='jet')
plt.ion()

# get the data from hillenbrand
from loadHillenbrand import *

filenb = 330 # for bigdat
filenbt = mpl.mlab.find(timedatnames==bigdatnames[filenb])[0]

fs, data = loadHill(bigdatnames[filenb]) 

windowSizeInSeconds = 0.064

windowSizeInSamples = nextpow2(np.ceil(fs * windowSizeInSeconds))
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
nFmax = np.int32(np.ceil(FHzmax/fs*NFT))
ytickslab = np.array([1000, 2000, 3000, 4000])
ytickspos = np.int32(np.ceil(ytickslab/fs*NFT))
fontsize = 16
figheight=4.5
figwidth=9.0

plt.figure(3)#, figsize=[figwidth, figheight])
plt.imshow(db(SX))
plt.hold(True)
for n in range(3):
    plt.plot(timeStampsInFrame,
             bigdat[filenb,5+n+3*np.arange(8)]/np.double(fs)*NFT,
             'ok')

plt.axis('tight')
plt.draw()

# parameters for Source and filter parts:
minF0 = 80
maxF0 = 500
stepNotes = 16 # this is the number of F0s within one semitone
# number of chirped spectral shapes between each F0
# this feature should be further studied before
# we find a good way of doing that.
chirpPerF0 = 1
formantsRange = {}
formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
formantsRange[1] = [ 550.0, 3500.0]
formantsRange[2] = [1400.0, 4500.0]
formantsRange[3] = [2400.0, 6000.0] # adding one for full band
formantsRange[4] = [3300.0, 8000.0]
formantsRange[5] = [4500.0, 8000.0]
formantsRange[6] = [5500.0, 8000.0]

# generate the autoregressive spectral dictionary:
bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = genARbasis(
    F, NFT, fs, maxF0=2*maxF0, formantsRange=formantsRange)
numberOfFormants = freqRanges.shape[0]
numberOfAmpPerFormantFreq = bwRange.size

WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
Fwgamma, Nwgamma = WGAMMA.shape

nbElPerF = Nwgamma/numberOfFormants

# generate the spectral combs for the source:
F0Table, WF0 = \
         generate_WF0_chirped(minF0, maxF0, fs, Nfft=NFT, \
                              stepNotes=stepNotes, \
                              lengthWindow=windowSizeInSamples, Ot=0.5, \
                              perF0=chirpPerF0, \
                              depthChirpInSemiTone=.15,
                              loadWF0=False,
                              analysisWindow='hanning')

WF0 = WF0[0:F, :] # ensure same size as SX 
NF0 = F0Table.size # number of harmonic combs
# Normalization:
WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))

# to add the unvoiced element:
withExtraUnvoiced = True
if withExtraUnvoiced:
    WF0 = np.concatenate((WF0, np.ones([F,1])), axis=1)
    NF0 = NF0 + 1

# putting the matrices together to make the required spectral dictionaries
# in order to use ARPIMM.SFSNMF:
W = [WF0.copy()]
nElPerFor = Nwgamma/numberOfFormants
for p in range(numberOfFormants):
    W.append(np.hstack((WGAMMA[:,(p*nElPerFor):((p+1)*nElPerFor)],
                        np.atleast_2d(np.ones(F)).T)))
    # adding a vector of ones, this is to "deactivate"
    # the corresponding W...

WR = generateHannBasis(numberFrequencyBins=F,
                       sizeOfFourier=NFT, Fs=fs,
                       frequencyScale='linear',
                       numberOfBasis=20, 
                       overlap=.75)

P = len(W)

# SFSNMF algorithm to estimate the parameters for the given signal:
niter = 80
G, GR, H, recoError2 = ARPIMM.SFSNMF(SX, W, WR, poleFrq=None, H0=None, \
                                     stepNotes=None, nbIterations=niter,
                                     dispEvol=True)

eps = 10**-50
Shat = [np.dot(W[0], G[0])]
mu = [np.dot(np.arange(NF0-1) * \
             (np.arange(NF0-1, 0, -1))**2, \
             G[0][:-1,:]) / \
      np.dot((np.arange(NF0-1, 0, -1))**2,
             np.maximum(G[0][:-1,:], eps))]

for p in range(1,P):
    mu.append(np.dot(np.arange(nbElPerF) * \
                     (np.arange(nbElPerF, 0, -1))**2,
                     G[p][:-1,:]) / \
              np.dot((np.arange(nbElPerF, 0, -1))**2,
                     np.maximum(G[p][:-1,:], eps)))
    Shat.append(np.dot(W[p],G[p]))

F0seq = F0Table[np.int32(mu[0])]

SXhat = H * np.vstack(np.dot(WR,GR)) * np.prod(Shat, axis=0)

fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(211)
ax.imshow(db(SX))
plt.hold(True)
for n in range(3):
    ax.plot(timeStampsInFrame,
             bigdat[filenb,5+n+3*np.arange(8)]/np.double(fs)*NFT,
             'ok')
plt.axis('tight')
plt.draw()
ax2 = fig.add_subplot(212, sharex=ax, sharey=ax)
ax2.imshow(db(SXhat))
for n in range(3):
    ax2.plot(timeStampsInFrame,
             bigdat[filenb,5+n+3*np.arange(8)]/np.double(fs)*NFT,
             'ok', markerfacecolor='w', markeredgewidth=4)
    ax2.plot(F0seq/np.double(fs)*NFT, ls='--')
Fformant = []
for p in range(1,P):
    Fformant.append(poleFrq[np.int32((p-1)*nbElPerF+mu[p])])
    ax2.plot(Fformant[p-1]/np.double(fs)*NFT)
plt.axis('tight')
plt.draw()

if False:
    # making the pictures:
    FHzmax = 4000.0
    nFmax = np.int32(np.ceil(FHzmax/fs*NFT))
    ytickslab = np.array([1000, 2000, 3000, 4000])
    ytickspos = np.int32(np.ceil(ytickslab/fs*NFT))
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
    
    plt.legend(legendThing.values(), loc='lower left')
    plt.axis('tight')
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.06)
