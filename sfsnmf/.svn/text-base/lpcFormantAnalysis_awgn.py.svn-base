#! /usr/bin/env python

# compute the formant trajectories :
#     first try:
#         compute LPC coefficients
#         compute the poles of corresponding filter
#         get the 4 poles with highest peak, avoid duplicates?

import time

import numpy as np

# import scikits.audiolab as al

import scikits.talkbox as tb

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.aspect'] = 'auto'

from matplotlib.pyplot import imshow as imageM

# from imageTools import imageM

import scipy
import scipy.signal

mpl.interactive(True)

def nextpow2(i):
    """
    Find 2^n that is equal to or greater than.
    
    code taken from the website:
    http://www.phys.uu.nl/~haque/computing/WPark_recipes_in_python.html
    """
    n = 2
    while n < i:
        n = n * 2
    return n


def db(val):
    """
    db(positiveValue)
    
    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(val)


def stlpc(longSignal, order=10, windowLength=1024, hopsize=512, axis=-1):
    """Compute "Short Time LPC":
          Cut the input signal in frames
          Compute the LPC on each of the frames
    
    """
    lengthSignal = longSignal.size
    
    currentWindow = np.zeros([windowLength,])
    
    Nb_windows = np.ceil((lengthSignal - windowLength) / (np.double(hopsize)) + 1.0)
    STLpc = np.ones([order + 1, Nb_windows])
    
    rootLpc = np.zeros([order, Nb_windows], dtype=np.complex)
    freqLpc = np.ones([(order - 2.0)/2.0, Nb_windows])
    specFromLpc = np.zeros([windowLength / 2.0 + 1, Nb_windows])
    sigmaS = np.zeros([Nb_windows, ])
    
    b_preamp=np.array([1.0,-0.99])
    a_preamp=np.array([1.0])
    longSignalPreamp = scipy.signal.lfilter(b_preamp,a_preamp,longSignal)
    
    for n in np.arange(Nb_windows):
        beginFrame = n * hopsize
        endFrame = np.minimum(n * hopsize + windowLength, lengthSignal)
        currentWindow[:endFrame-beginFrame] = longSignalPreamp[beginFrame: endFrame]
        currentWindow *= np.hamming(windowLength)
        STLpc[:,n], sigmaS[n], trash = tb.lpc(currentWindow, order)
        specFromLpc[:,n] = lpc2spec(STLpc[:,n], sigmaS[n], fs, windowLength)
        rootLpc[:,n] = np.roots(STLpc[:,n])
        freqLpcTmp = np.angle(rootLpc[:,n]) / (2.0 * np.pi) * fs
        freqLpcTmp = freqLpcTmp[freqLpcTmp>0.0]
        freqLpcTmp.sort()
        nbMinPositiveRoots = freqLpcTmp[0:(order - 2.0)/2.0].size
        freqLpc[0:nbMinPositiveRoots,n] = freqLpcTmp[0:(order - 2.0)/2.0]
        
    return STLpc, rootLpc, freqLpc, specFromLpc, sigmaS

def lpc2spec(lpc, sigma, fs, nfft):
    orderPlus1 = lpc.size
    matExp = np.exp(- 1j * np.outer(2 * np.pi * \
                                    np.arange(nfft / 2.0 + 1) / \
                                    np.double(nfft),
                                    np.arange(orderPlus1, dtype=np.double)))
    return sigma / np.abs(np.dot(matExp, lpc)) 

def hz2mel(f):
    return 2595 * np.log10(1+f/700.0)

## longSignal, fs, enc = al.oggread('/home/durrieu/work/BDD/EN/00000000/0201402D/00000001.ogg')
## longSignalFinal, fs, enc = al.oggread('/home/durrieu/work/BDD/EN/00000003/0201402D/00000001.ogg')
##longSignal, fs, enc = al.oggread('/users/jeanlouis/work/BDD/speedlingua/EN/before/0201402D/00000001.ogg')
##longSignalFinal, fs, enc = al.oggread('/users/jeanlouis/work/BDD/speedlingua/EN/after/0201402D/00000001.ogg')

# longSignal, fs, enc = al.wavread('/Users/jeanlouis/work/BDD/hillenbrand/vowels/kids/b13ah.wav')

order=14
#windowLength=512.0
windowLength=1024.
#windowLength=256.
NFT=windowLength*1.0
#hopsize=128.0
hopsize=256.
#hopsize=128.
axis=-1

from loadHillenbrand import *

if True:
    dirSaveArx = str('').join([prefixBDD,
                               '/result_LPCOrder-', str(order),
                               '_winSize-',str(windowLength),
                               '_hopsize-',str(hopsize),
                               '/'])
    fileSaveParam = str('').join([dirSaveArx,
                                  '/commonParameters.npz'])
    
    if not(os.path.isdir(dirSaveArx)):
        os.mkdir(dirSaveArx)
        
    np.savez(fileSaveParam, order=order,
             windowLength=windowLength,
             hopsize=hopsize)
    
    nbFilesHill = timedat.shape[0]
    
    numberGTFormants = 3
    numberOfFormants = 6
    
    snrs = (
        -10., -5., 0., 5., 10., 20., 30)
    
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
    ## mpl.mlab.find(bigdatnames=='m07oa')[0] # 1163
    ## ae : 100
    ## oa : 1163
    for filenb in range(nbFilesHill):
        ##filenb = 100 # for bigdat
        for snrnb, snr in enumerate(snrs):#if True or bigdatnames[filenb][:3] == 'm01':
            filenbt = mpl.mlab.find(timedatnames==bigdatnames[filenb])[0]
            print "processing file %s, number %d of %d" %(bigdatnames[filenb],
                                                          filenb,
                                                          nbFilesHill)
            
            fs, data = loadHill(bigdatnames[filenb])
            Fs = fs
            data = data / 2.0**(data[0].nbytes * 8.0 - 1) # -1 because signed int.
            
            # adding noise:
            print 'snr', snr, 'effective', 10*np.log10(
                (data**2).sum()
                /(1.*(np.sqrt(data.var() * (10 ** (- snr / 10.)))
                      * np.random.randn(data.size))**2).sum()
                )
            data += (
                np.sqrt(data.var() * (10 ** (- snr / 10.)))
                * np.random.randn(data.size))
            
            stLpc, rootLpc, freqLpc, specFromLpc, sigmaS = stlpc(data, order=order,
                                                                 windowLength=windowLength,
                                                                 hopsize=hopsize)
            
            timeInFrame = 0.001 * timedat[filenbt] / np.double(hopsize/np.double(fs))
            timeStampsInFrame = timeInFrame[0] + \
                                (timeInFrame[1]-timeInFrame[0]) * \
                                np.arange(1,9) * 0.1# 10%... 80%
            
            time0 = time.time()
            
            # draw figure plus formant annotated
            FHzmax = 4000.0
            nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
            ytickslab = np.array([1000, 2000, 3000, 4000])
            ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
            fontsize = 16
            figheight=4.5
            figwidth=9.0
            
            displayEvolution = False #True
            
            if displayEvolution:
                plt.figure(1)#, figsize=[figwidth, figheight])
                plt.clf()
                imageM(db(specFromLpc))
                plt.hold(True)
                
            trueFormant = {}
            for n in range(3):
                trueFormant[n] = bigdat[filenb,5+n+3*np.arange(8)]
                if displayEvolution:
                    plt.plot(timeStampsInFrame,
                             trueFormant[n]/np.double(fs)*NFT,
                             'ok')
            
            if displayEvolution:
                plt.plot(freqLpc.T/np.double(fs)*NFT, )
                plt.axis('tight')
                plt.draw()
                
            compTimes[snrnb + filenb*len(snrs)] = time.time() - time0
            
            FP = hz2mel(freqLpc)
            for t in range(8): # there are 8 GT values per formant and per file
                for n in range(numberGTFormants):
                    errorMatrix[snr][n,:,8*filenb+t] = (
                        FP[:,timeStampsInFrame[t]] - 
                        hz2mel(trueFormant[n][t]))
                    
            erFilMatrix[snr][:,:,filenb] = np.sqrt(
                np.mean((errorMatrix[snr][:,:,8*filenb+np.arange(8)])**2,
                        axis=2))
            
            print compTimes[snrnb + filenb*len(snrs)], "sec. for ", erFilMatrix[snr][:,:,filenb]
            
            fileSaveArx = str('').join([dirSaveArx,
                                        '/', bigdatnames[filenb],
                                        '_lpc.npz'])
            np.savez(fileSaveArx, trueFormant=trueFormant,
                     timeStampsInFrame=timeStampsInFrame,
                     stLpc=stLpc, rootLpc=rootLpc,
                     freqLpc=freqLpc, specFromLpc=specFromLpc, sigmaS=sigmaS)

            np.savez(fileSaveParam, order=order,
                     windowLength=windowLength,
                     hopsize=hopsize,
                     erFilMatrix=erFilMatrix,
                     errorMatrix=errorMatrix,
                     compTimes=compTimes,
                     numberOfFormants=numberOfFormants,
                     numberGTFormants=numberGTFormants)
