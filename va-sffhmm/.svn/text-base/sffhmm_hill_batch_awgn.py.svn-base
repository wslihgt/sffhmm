"""sffhmm_hill_batch

This is the script that computes the results for the article:
    J.-L. Durrieu and J.-Ph. Thiran,
    'Source/Filter Factorial Hidden Markov Model,
    with Application to Pitch and Formant Tracking,'
    submitted, 2012.

Script to run a batch processing on all the files from the
Hillenbrand dataset, available online at the address:
    http://homepages.wmich.edu/~hillenbr/voweldata.html



Dependencies:
    * numpy, scipy (for wav file I/O)
    * matplotlib 
    * sffhmm.py: as provided with our article
    * scikits.learn: v. 0.8, 0.9 or 0.10. lower or higher versions
      may lead to problems with the `hmm` module. 
    * Hillenbrand's vowel dataset and loadHillenbrand.py (provided)
"""

import sffhmm as fhmm
import scipy.io.wavfile as wav
import numpy as np
import time

#     set verbose to True for more (debugging) outputs from the
#     program
verbose = False # True # True
debug = False # True 

####################
# USEFUL FUNCTIONS #
####################
def mel2hz(f):
    """fhz = mel2hz(fmel)
    
    converts fmel, expressed in mel, into fhz, expressed in Hz.
    """
    return 700.0 * (10**(f / 2595.0) - 1)

def hz2mel(f):
    """fmel = hz2mel(fhz)
    
    converts fhz, expressed in Hz, into fmel, expressed in mel.
    """
    return 2595 * np.log10(1+f/700.0)

####################
# Configuration    #
####################
# for display purposes, comment if not wanted.
import matplotlib.pyplot as plt
from matplotlib.mlab import find as mlabfind

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

import sys
from loadHillenbrand import *

pathToArx = prefixBDD

resDir = os.listdir(pathToArx+'/kids/')
resDir.extend(os.listdir(pathToArx+'/men/'))
resDir.extend(os.listdir(pathToArx+'/women/'))

nbFilesHill = timedat.shape[0]
numberGTFormants = 3

vowels = np.array([n[-2:] for n in bigdatnames])
vow = ['ae', 'ah', 'aw', 'eh', 'ei', 'er',
       'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


windowSizeInSeconds = 0.032 # 0.064 # 
fs = 16000
lenWin = fhmm.nextpow2(np.ceil(fs * windowSizeInSeconds))
steWin = lenWin/4
NFT = lenWin
maxFinFT = 8000
F = np.ceil(maxFinFT * NFT / np.double(fs))

chirpPerF0 = 1
ninner = 3 # 50
# DISPLAY:
#     set dispMat to False if you do not want to have the results displayed
dispMat = True
methPostInit = 'triangle' # 'lms'
withNoiseFloor = True

formantsRange = {}
formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
formantsRange[1] = [ 550.0, 3500.0]
formantsRange[2] = [1400.0, 4500.0]
formantsRange[3] = [2400.0, 6000.0] # adding one for full band
formantsRange[4] = [3300.0, 8000.0]
#formantsRange[4] = [3300.0, 7000.0]
formantsRange[5] = [4500.0, 8000.0]
formantsRange[6] = [5500.0, 8000.0]
# formantsRange[7] = [6500.0, 8000.0]

numberOfFormants = len(formantsRange)

# instantiating the SFFHMM object:
sffhmm = fhmm.SFFHMM(samplingRate=fs,NFT=NFT,withNoiseF0=True,
                     withFlatFilter=True,
                     formantRanges=formantsRange,F0range=[80,500],
                     chirpPerF0=chirpPerF0,stepNotes=16,
                     withNoiseFloor=withNoiseFloor)
# and generating the transition probability matrix:
sffhmm.generateTransMatForSFModel(stepSize=steWin/np.double(fs))

# the following file will be written, in which the results 
# for all the files are saved:
import datetime
currentTime = datetime.datetime.strftime(\
        datetime.datetime.now(), format='%Y%m%dT%H%M')

savefilename = (
    'errorMatHillenbrand_SNRs_ninner-'+str(ninner)+
    '_nbFP-'+str(numberOfFormants)+
    '_chirpF0-'+str(chirpPerF0)+
    '_lenWin-'+str(lenWin)+
    '_steWin-'+str(steWin)+
    '_'+currentTime+
    '.npz')

snrs = (
    -10., -5., 0., 5., 10., 20., 30)

# keeping results in memory:
erFilMatrix = {}
errorMatrix = {}
for snr in snrs:
    erFilMatrix[snr] = np.ones([numberGTFormants,
                                numberOfFormants,
                                nbFilesHill]) # unconstrained
    
    errorMatrix[snr] = np.ones([numberGTFormants,
                                numberOfFormants,
                                8*nbFilesHill]) # unconstrained

compTimes = np.zeros(nbFilesHill * len(snrs))
############################
# MAIN LOOP OVER ALL FILES #
############################
for filenb in range(0,nbFilesHill):
    # Loading the data, with the variables specific to Hillenbrand dataset:
    filenbt = mlabfind(timedatnames==bigdatnames[filenb])[0]
    
    for snrnb,snr in enumerate(snrs):
        fs, data = loadHill(bigdatnames[filenb])
        data = np.double(data)
        data /= (abs(data)).max()
        
        # adding noise:
        print 'snr', snr, 'effective', 10*np.log10(
            (data**2).sum()
            /(1.*(np.sqrt(data.var() * (10 ** (- snr / 10.)))
                  * np.random.randn(data.size))**2).sum()
            )
        data += (
            np.sqrt(data.var() * (10 ** (- snr / 10.)))
            * np.random.randn(data.size))
        # Computing the time-frequency representation:
        
        time0 = time.time()
        
        obs, freqs, timeStamps = fhmm.stft(data, fs=fs, window=fhmm.hann(lenWin),
                                           hopsize=steWin, nfft=NFT)
        # As stated in the article, the model applies to the log-spectrum:
        obs2 = np.log(np.array(abs(obs.T)**2, order='C'))
        
        # thresholding to avoid Inf values: 
        eps = obs2[obs2>-np.Inf].min()
        obs2 = np.maximum(obs2, eps)
        
        # Getting the formant ground truth from Hillenbrand's annotations:
        timeInFrame = 0.001 * timedat[filenbt] / np.double(steWin/np.double(fs))
        timeStampsInFrame = timeInFrame[0] + \
                            (timeInFrame[1]-timeInFrame[0]) * \
                            np.arange(1,9) * 0.1# 10%... 80%
        
        trueFormants = {}
        for n in range(3):
            trueFormants[n] = bigdat[filenb,5+n+3*np.arange(8)]
            
        # Estimating the parameters and posterior probabilities for the given
        # file: 
        mpost, states, posteriors = sffhmm.decode_var(obs=obs2,
                                                      n_innerLoop=ninner,
                                                      verbose=verbose, debug=debug,
                                                      thresholdEnergy=0,
                                                      postInitMeth=methPostInit)
        # Estimating the corresponding energy and recording condition parameters:
        logH = sffhmm._compute_energyComp(obs=obs2,
                                          posteriors=posteriors,
                                          debug=False)
        rfilter = sffhmm._compute_recording_condition(obs=obs2,
                                                      posteriors=posteriors,
                                                      logH=logH,
                                                      debug=False)
        # The corresponding a posteriori mean (approximated spectrum)
        # is given by:
        mpost += logH + rfilter
        
        # Converting the state sequences (i.e. the estimated F0 and formant
        # tracks) into values in "number of frequency bins" and "mel"
        statesInNF = {}
        statesInMel = np.zeros([len(sffhmm.n_states), obs2.shape[0]])
        statesInNF[0] = sffhmm.F0Table[states[0]/chirpPerF0] / np.double(fs) * NFT
        statesInMel[0] = hz2mel(sffhmm.F0Table[states[0]/chirpPerF0]) 
        for n in range(sffhmm.numberOfFormants):
            idx1 = (n) * sffhmm.nElPerFor
            idx2 = (n+1) * sffhmm.nElPerFor
            freqs = np.concatenate([sffhmm.poleFrq[idx1:idx2], [0]])
            statesInNF[n+1] = freqs[states[n+1]] / \
                              np.double(fs) * NFT
            statesInMel[n+1] = hz2mel(freqs[states[n+1]])
            
        compTimes[snrnb + filenb*len(snrs)] = time.time() - time0
        
        # Displaying the resulting tracks, showing also the ground truth:
        if dispMat:
            fig = plt.figure(10)
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax1.imshow(obs2.T)
            ax1.set_title(bigdatnames[filenb]+': observation')
            for n in range(sffhmm.numberOfFormants+1):
                ax1.plot(statesInNF[n], '.-')
            for n in range(3): # Ground Truth
                ax1.plot(timeStampsInFrame,
                         trueFormants[n]/np.double(fs)*NFT,
                         'ok')
            ax1.axis('tight')
            ax2=fig.add_subplot(212, sharex=ax1,sharey=ax1)
            ax2.imshow(mpost.T)
            ax2.set_title('posterior mean')
            for n in range(sffhmm.numberOfFormants+1):
                ax2.plot(statesInNF[n], '.-')
            for n in range(3): # Ground Truth
                ax2.plot(timeStampsInFrame,
                         trueFormants[n]/np.double(fs)*NFT,
                         'ok')
            ax2.axis('tight')
            fig.canvas.draw()
        # Computing the estimation errors: 
        for t in range(8):
            for n in range(numberGTFormants):
                errorMatrix[snr][n,:,8*filenb+t] = \
                    (statesInMel[1:,np.rint(timeStampsInFrame[t])]-\
                     hz2mel(trueFormants[n][t]))
        erFilMatrix[snr][:,:,filenb] = np.sqrt( \
                np.mean((errorMatrix[snr][:,:,8*filenb+np.arange(8)])**2,
                        axis=2))
        
        # print and save the results:
        print compTimes[snrnb + filenb*len(snrs)], "sec. for ",
        print erFilMatrix[snr][:,:,filenb][[range(3), range(3)]]
        np.savez(savefilename,
                 errorMatrix=errorMatrix,
                 erFilMatrix=erFilMatrix,
                 compTimes=compTimes)


########################################
### STATISTICS AND MORE ON THE RESULTS #
########################################
##fontsize = 12

### parameters for drawing the histogram of errors, for each formant:
##nbins = 100
##arrayOfHistBins = np.zeros([numberGTFormants*numberOfFormants, nbins+1])
##arrayOfHist = np.zeros([numberGTFormants*numberOfFormants, nbins])

### Statistics over all the files: the following matrices summarize
### which formant track best estimates which formant ground truth.
### 
##errorMeans = np.zeros([numberGTFormants, numberOfFormants])
##errorMedians = np.zeros([numberGTFormants, numberOfFormants])
##errorStdev = np.zeros([numberGTFormants, numberOfFormants])

### gathering and averaging all the results, for the mean error values
### for each file:
##for n in range(numberGTFormants):
##    for e in range(numberOfFormants):
##        arrayOfHist[n*numberOfFormants+e], \
##            arrayOfHistBins[n*numberOfFormants+e] = \
##                np.histogram((errorMatrix[n,e,:]), bins=nbins)
        
##        ##print 'mean errors: GT formant ', n, \
##        ##      ' est. formant ', e, ' uncst: ', \
##        errorMeans[n,e] =     np.sqrt(errorMatrix[n,e,:]**2).mean()
##        errorMedians[n,e] =   np.median(np.sqrt(errorMatrix[n,e,:]**2))
##        errorStdev[n,e] =     np.sqrt(errorMatrix[n,e,:]**2).std()

##centerValHist = (arrayOfHistBins[:,:100] + arrayOfHistBins[:,1:]) * .5 

### Displaying these errors: 
##figwidth, figheight = [12,9]

##plt.figure(figsize=[figwidth, figheight])
### For each formant ground truth (each line), plot the following
### estimated formant tracks:
##rangeGTFP = np.array([[0, 1, 2, 3, 4], # gt formant 1 <-> est. formant 2 and 3
##                      [0, 1, 2, 3, 4], # gt formant 2 <-> est. formant 3 and 4 
##                      [0, 1, 2, 3, 4]]) # formant 3 <-> formant 5

##ylimMax = 5000
##for n in range(numberGTFormants):
##    plt.subplot(1,numberGTFormants,n+1)
##    plt.plot(centerValHist[n*numberOfFormants+rangeGTFP[n]].T,
##             arrayOfHist[n*numberOfFormants+rangeGTFP[n]].T)
##    plt.xlim([-1500, 1500])
##    plt.ylim([0, ylimMax])
##    plt.title('F%d and F%d <-> GTF%d' %(rangeGTFP[n][0]+1,
##                                        rangeGTFP[n][1]+1, n+1))
##    if n==0:plt.ylabel('Var-SFFHMM')

##plt.subplots_adjust(left  =.07,
##                    right =.96,
##                    bottom=.07,
##                    top   =.96)

##plt.savefig(savefilename+'_histoError.pdf')

### Statistics for individual estimated values
### (results not averaged within the files):
##nbins = 100
##arFilOfHistBins = np.zeros([numberGTFormants*numberOfFormants, nbins+1])
##arFilOfHist = np.zeros([numberGTFormants*numberOfFormants, nbins])

##errorFileMeans = np.zeros([numberGTFormants, numberOfFormants])
##errorFileMedians = np.zeros([numberGTFormants, numberOfFormants])
##errorFileStdev = np.zeros([numberGTFormants, numberOfFormants])
##for n in range(numberGTFormants):
##    for e in range(numberOfFormants):
##        arFilOfHist[n*numberOfFormants+e], \
##            arFilOfHistBins[n*numberOfFormants+e] = \
##                np.histogram((erFilMatrix[n,e,:]), bins=nbins)
        
##        ##print 'mean errors: GT formant ', n, \
##        ##      ' est. formant ', e, ' uncst: ', \
##        errorFileMeans[n,e] =     (erFilMatrix[n,e,:]).mean()
##        errorFileMedians[n,e] =   np.median(erFilMatrix[n,e,:])
##        errorFileStdev[n,e] =     (erFilMatrix[n,e,:]).std()


##cenFilValHist = (arFilOfHistBins[:,:100] + arFilOfHistBins[:,1:]) * .5 

##plt.figure(figsize=[figwidth, figheight])
###linestylelist = ('-', '--', ':', '-.')
##rangeGTFP = np.array([[0, 1, 2, 3, 4], # gt formant 1 <-> est. formant 2 and 3
##                      [0, 1, 2, 3, 4], # gt formant 2 <-> est. formant 3 and 4 
##                      [0, 1, 2, 3, 4]]) # formant 3 <-> formant 5
##for n in range(numberGTFormants):
##    plt.subplot(1,numberGTFormants,n+1)
##    plt.plot(cenFilValHist[n*numberOfFormants+rangeGTFP[n]].T, 
##             arFilOfHist[n*numberOfFormants+rangeGTFP[n]].T, )
##    plt.xlim([0,1000])
##    plt.title('F%d and F%d <-> GTF%d' %(rangeGTFP[n][0]+1,
##                                        rangeGTFP[n][1]+1, n+1))
##    if n==0:plt.ylabel('Var-SFFHMM')

##plt.subplots_adjust(left  =.05,
##                    right =.96,
##                    bottom=.07,
##                    top   =.96)

##plt.savefig(savefilename+'_histoErFil.pdf')

##if True:
##    fontsize=16
##    plt.figure()
##    plt.title('VAR-SFFHMM', fontsize=fontsize)
##    plt.boxplot(erFilMatrix[range(3), range(3),:].T,
##                whis=1., sym='b.')
##    plt.gcf().get_axes()[0].set_yscale('log')
##    plt.ylim([1,5000])
##    plt.xticks(fontsize=fontsize)
##    plt.yticks(fontsize=fontsize)
##    plt.savefig(savefilename+'_boxplotLog.pdf')

##import scipy.stats.stats as stats

##quantiles = np.zeros([3*3])

##for n in range(3):
##    for q in range(3):
##        quantiles[3*n+q] = stats.scoreatpercentile(erFilMatrix[n,n,:],
##                                                   per=25+q*25)

##print "quantiles:", quantiles
