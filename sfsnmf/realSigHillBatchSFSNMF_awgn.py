#!/usr/bin/python

"""Experiments for the SFSNMF algorithm
Description
-----------
Runs the experiments for the SFSNMF algorithm, for various SNR cconditions,
as described in:

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
 2) run the script. This script may run for a long time until it finishes
    all the computations.
    
2013 - Jean-Louis Durrieu (http://www.durrieu.ch/research/)

"""

import numpy as np
import os.path
import time
import ARPIMM

import scipy.linalg as spla

##import scikits.audiolab as al

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.aspect'] = 'auto'

from tracking import viterbiTrackingArray

# SOME USEFUL, INSTRUMENTAL, FUNCTIONS # 
from SFSNMF_functions import *

displayEvolution = True

import matplotlib.pyplot as plt
plt.rc('image',cmap='jet')
plt.ion()

minF0 = 80
maxF0 = 500
stepNotes = 16 # this is the number of F0s within one semitone
K = 10 # number of spectral shapes for the filter part
R = 0 # number of spectral shapes for the accompaniment
P = 30 # number of elements in dictionary of smooth filters

# number of chirped spectral shapes between each F0
# this feature should be further studied before
# we find a good way of doing that.
chirpPerF0 = 1

#########################################################################
# LOAD HILLENBRAND VARIABLES: MODIFY THAT FILE TO MAKE THIS SCRIPT WORK #
#########################################################################
from loadHillenbrand import *

# PARAMETERS #
fs=16000.0

windowSizeInSeconds = 0.064 # 0.032

windowSizeInSamples = nextpow2(np.ceil(fs * windowSizeInSeconds))
hopsize = windowSizeInSamples/4
NFT = windowSizeInSamples
maxFinFT = 8000
F = np.ceil(maxFinFT * NFT / np.double(fs))
Ot = .6
numberOfAmpsPerPole = 10
numberOfFreqPerPole = 40
niterSpSm = 80
numberGTFormants = 3

formantsRange = {}
formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
formantsRange[1] = [ 550.0, 3500.0]
formantsRange[2] = [1400.0, 4500.0]
formantsRange[3] = [2400.0, 6000.0] # adding one for full band
formantsRange[4] = [3300.0, 8000.0]
formantsRange[5] = [4500.0, 8000.0]
formantsRange[6] = [5500.0, 8000.0]

numberOfFormants = len(formantsRange)

##############################################
# GENERATING THE SPECTRAL SHAPE DICTIONARIES # 
##############################################
bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = (
         genARbasis(F, NFT, fs, maxF0=maxF0,
                    formantsRange=formantsRange,
                    numberOfAmpsPerPole=numberOfAmpsPerPole,
                    numberOfFreqPerPole=numberOfFreqPerPole))
numberOfFormants = freqRanges.shape[0]
numberOfAmpPerFormantFreq = bwRange.size

poleFrqMel = hz2mel(poleFrq)

WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
Fwgamma, Nwgamma = WGAMMA.shape

nbElPerF = Nwgamma/numberOfFormants

F0Table, WF0 = \
         generate_WF0_chirped(minF0, maxF0, fs, Nfft=NFT, \
                              stepNotes=stepNotes, \
                              lengthWindow=windowSizeInSamples, Ot=Ot, \
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

#################################################################
# PUTTING THE DICTIONARIES INTO SHAPE FOR THE SFSNMF ALGORITHMS #
#################################################################
W = {}
W[0] = WF0.copy()
nElPerFor = Nwgamma/numberOfFormants
for p in range(numberOfFormants):
    W[p+1] = np.hstack((WGAMMA[:,(p*nElPerFor):((p+1)*nElPerFor)],
                        np.atleast_2d(np.ones(F)).T))
    # adding a vector of ones, this is to "deactivate"
    # the corresponding W...

numberOfBasisWR = 20
WR = generateHannBasis(numberFrequencyBins=F,
                       sizeOfFourier=NFT, Fs=fs,
                       frequencyScale='linear',
                       numberOfBasis=numberOfBasisWR, 
                       overlap=.75)

P = len(W)

#########################################
# THE ARCHIVE WHERE TO SAVE THE RESULTS # 
#########################################
import datetime
currentTime = datetime.datetime.strftime(\
    datetime.datetime.now(), format='%Y%m%dT%H%M')

dirSaveArx = str('').join([prefixBDD,
                           '/result_Ot-', str(Ot),
                           '_nbFP-', str(numberOfFormants),
                           '_numberOfBasisWR-', str(numberOfBasisWR),
                           '_numberOfFreqPerPole-%d' %numberOfFreqPerPole,
                           '_numberOfAmpsPerPole-%d' %numberOfAmpsPerPole, 
                           '_winSize-',str(windowSizeInSamples),
                           '_hopsize-',str(hopsize),
                           '_niterSpSm-',str(niterSpSm),
                           '_', currentTime,
                           '/'])

if not(os.path.isdir(dirSaveArx)):
    os.mkdir(dirSaveArx)

fileSaveParam = str('').join([dirSaveArx,
                              '/commonParameters.npz'])+

kwParams = {
    'Ot':Ot, 'W':W, 'F0Table':F0Table, 'WGAMMA':WGAMMA,
    'poleFrq':poleFrq, 'poleAmp':poleAmp, 
    'formantsRange':formantsRange,
    'numberOfFormants':numberOfFormants, 
    'numberOfAmpsPerPole':numberOfAmpsPerPole,
    'numberOfFreqPerPole':numberOfFreqPerPole,
    'niter': niterSpSm}

np.savez(fileSaveParam, **kwParams)

nbFilesHill = timedat.shape[0]

# define the tested SNRs
snrs = (
    -10., -5., 0., 5., 10., 20., 30)
#snrs = (
#    20., )

erFilMatrix = {}
errorMatrix = {}
for snr in snrs:
    erFilMatrix[snr] = np.ones([numberGTFormants,
                                numberOfFormants,
                                nbFilesHill])

    errorMatrix[snr] = np.ones([numberGTFormants,
                                numberOfFormants,
                                8*nbFilesHill])

compTimes = np.zeros(nbFilesHill * len(snrs))

###########################################################
# LOOP OVER ALL SNR CONDITIONS AND FILES FROM HILLENBRAND #
###########################################################
nbFilesHill = timedat.shape[0]

# to find the file number of a given file:
## mpl.mlab.find(bigdatnames=='m07oa')[0] # 1163
## ae : 100
## oa : 1163

for filenb in range(nbFilesHill):
    filenbt = mpl.mlab.find(timedatnames==bigdatnames[filenb])[0]
    print "processing file %s, number %d of %d" %(bigdatnames[filenb],
                                                  filenb,
                                                  nbFilesHill)
    
    for snrnb, snr in enumerate(snrs):
        # load the data
        fs, data = loadHill(bigdatnames[filenb])
        data = data / \
               2.0**(data[0].nbytes * \
                     8.0 - 1) # -1 because signed int.
        
        # adding noise
        data += (
            np.sqrt(data.var() * (10 ** (- snr / 10.)))
            * np.random.randn(data.size))
        
        timeInFrame = 0.001 * timedat[filenbt] / \
                      np.double(hopsize/np.double(fs))
        timeStampsInFrame = timeInFrame[0] + \
                            (timeInFrame[1]-timeInFrame[0]) * \
                            np.arange(1,9) * 0.1# 10%... 80%
        
        time0 = time.time()
        X = stft(data, fs=fs, hopsize=hopsize,
                 window=np.hanning(windowSizeInSamples), nfft=NFT)
        
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
        
        # display 'static' figures, after each estimation
        #     good way to control what was estimated, but increases
        #     computation time
        displayStatic = False #True #
        # display 'evolution' figures, during the estimation
        #     warning: this may drastically increase the computation time 
        displayEvolution = False #False #True#
        
        print "Running SFSNMF:"
        G, GR, H, recoError1 = ARPIMM.SFSNMF(SX, W, WR,
                                             poleFrq=None, H0=None, 
                                             stepNotes=None,
                                             nbIterations=niterSpSm,
                                             verbose=False,
                                             dispEvol=displayEvolution)
        
        if displayStatic:
            plt.figure(3)
            plt.clf()
            plt.title(bigdatnames[filenb])
            plt.imshow(db(SX))
            clmt = plt.figure(3).get_axes()[0].get_images()[0].get_clim()
            plt.hold(True)

        # THE GROUND TRUTH FORMANTS
        # as annotated in Hhillenbrand's files
        trueFormant = {}
        for n in range(3):
            trueFormant[n] = bigdat[filenb,5+n+3*np.arange(8)]
            if displayStatic:
                plt.plot(timeStampsInFrame,
                         trueFormant[n]/np.double(fs)*NFT,
                         'ok')
                
        if displayStatic:
            plt.axis('tight')
            plt.draw()
            
        eps = 10**-50
        if displayStatic:
            Shat = [np.dot(W[0], G[0])]

        # EXTRACT THE ESTIMATED SEQUENCES OF FORMANTS
        # from the estimated matrices output by the SFSNMF algorithm
        mu = {}
        mu[0] = np.dot(np.arange(NF0-1) * \
                       (np.arange(NF0-1, 0, -1))**2, \
                       G[0][:-1,:]) / \
                       np.dot((np.arange(NF0-1, 0, -1))**2,
                              np.maximum(G[0][:-1,:], eps))
        
        for p in range(1,P):
            mu[p] = np.dot(np.arange(nbElPerF) * \
                           (np.arange(nbElPerF, 0, -1))**2,
                           G[p][:-1,:]) / \
                           np.dot((np.arange(nbElPerF, 0, -1))**2,
                                  np.maximum(G[p][:-1,:], eps))
            if displayStatic:
                Shat.append(np.dot(W[p],G[p]))
                
        # The F0 frequency sequence, in Hz:
        F0seq = F0Table[np.int32(mu[0])]
        
        compTimes[snrnb + filenb*len(snrs)] = time.time() - time0
        
        if displayStatic:
            SXhat = H * np.vstack(np.dot(WR,GR)) * np.prod(Shat, axis=0)
            plt.figure(4)
            plt.clf()
            plt.title(bigdatnames[filenb])
            plt.imshow(db(SXhat))
            plt.clim(clmt)
            for n in range(3):
                plt.plot(timeStampsInFrame,
                         bigdat[filenb,5+n+3*np.arange(8)]/\
                         np.double(fs)*NFT,
                         'ok', markerfacecolor='w', markeredgewidth=4)
            plt.plot(F0seq/np.double(fs)*NFT, ls='--', lw=1)

        # The Formant frequencies, in Hz
        Fformant = {}
        for p in range(1,P):
            Fformant[p-1] = poleFrq[np.int32((p-1)*nbElPerF+mu[p])]
            if displayStatic: plt.plot(Fformant[p-1]/np.double(fs)*NFT)
            
        if displayStatic: plt.axis('tight');plt.draw()
        
        fileSaveArx = str('').join([dirSaveArx,
                                    '/', bigdatnames[filenb],
                                    '.npz'])
        
        # state formant frequencies in Mel:
        statesInMel = np.zeros([numberOfFormants, H.size])
        for p in range(1, numberOfFormants+1):
            nbElPerF = G[p].shape[0]-1 
            statesInMel[p-1] = poleFrqMel[\
                np.int32((p-1)*nbElPerF+mu[p])]

        ###############################
        # COMPUTING THE ERROR METRICS #
        ###############################
        for t in range(8):
            for n in range(numberGTFormants):
                errorMatrix[snr][n,:,8*filenb+t] = \
                    (statesInMel[:,np.rint(timeStampsInFrame[t])]-\
                     hz2mel(trueFormant[n][t]))
        erFilMatrix[snr][:,:,filenb] = np.sqrt( \
            np.mean((errorMatrix[snr][:,:,8*filenb+np.arange(8)])**2,
                    axis=2))
        
        print compTimes[snrnb + filenb*len(snrs)], "sec. for ",
        print erFilMatrix[snr][np.arange(3),np.arange(3),filenb]
        
        np.savez(fileSaveArx, trueFormant=trueFormant,
                 Fformant=Fformant,
                 H=H, GR=GR, G=G,
                 recoError1=recoError1,
                 F0seq=F0seq, 
                 filenb=filenb,
                 timeStampsInFrame=timeStampsInFrame, mu=mu)
        
        np.savez(fileSaveParam, 
                 erFilMatrix=erFilMatrix,
                 errorMatrix=errorMatrix,
                 compTimes=compTimes,
                 **kwParams)

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
    plt.xlabel('Frequency (Hz)', fontsize=fontsize)
    plt.legend((['F0 = %dHz' %(F0Table[nbF0])]))
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
    
    ##plt.rc('text', usetex=False)
    plt.figure(figsize=[figwidth, figheight])
    lsRange = ('-', '--', ':')
    N1 = 500
    #N2 = N1+12
    step = 4
    nbElts = 4
    legendThing = {}
    for nn in range(nbElts):
        
        plt.plot(db(WGAMMA[:nFmax,N1+nn*step]), lsRange[(nn+N1)%len(lsRange)],
                 color='k')
        plt.xticks(ytickspos, ytickslab, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        legendThing[nn-N1] = u'%1.3f' %(poleAmp[N1+nn*step])
    
    plt.legend(legendThing.values(), loc='best')
    plt.xlabel('Frequency (Hz)', fontsize=fontsize)
    plt.axis('tight')
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
