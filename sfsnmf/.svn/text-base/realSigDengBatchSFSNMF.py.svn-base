#!/usr/bin/python

import numpy as np
import os.path
# import SIMM
import time
import ARPIMM
import sys

sys.path.append('../tools')
import speechTools as st
import manipTools as mt

import scipy.linalg as spla

import warnings
import scikits.audiolab as al

##import scikits.audiolab as al

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.aspect'] = 'auto'

from tracking import viterbiTrackingArray

# SOME USEFUL, INSTRUMENTAL, FUNCTIONS

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

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)
    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return sum((-np.log(X / Y) + (X / Y) - 1))


# DEFINING SOME WINDOW FUNCTIONS

def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)
    
    Computes a "sinebell" window function of length L=lengthWindow
    
    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1
    """
    window = np.sin((np.pi * (np.arange(lengthWindow))) \
                    / (1.0 * lengthWindow))
    return window

def hann(args):
    """
    window = hann(args)
    
    Computes a Hann window, with NumPy's function hanning(args).
    """
    return np.hanning(args)

# FUNCTIONS FOR TIME-FREQUENCY REPRESENTATION

def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, \
         fs=44100.0):
    """
    X, F, N = stft(data, window=sinebell(2048), hopsize=1024.0,
                   nfft=2048.0, fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  : one-dimensional time-series to be
                                analyzed
        window=sinebell(2048) : analysis window
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation (the user has to provide an
                                even number)
        fs=44100.0            : sampling rate of the signal
        
    Outputs:
        X                     : STFT of data
        F                     : values of frequencies at each Fourier
                                bins
        N                     : central time at the middle of each
                                analysis window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(lengthWindow / 2.0),data))          
    lengthData = data.size
    
    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1  
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([newLengthData - lengthData])))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2.0 + 1
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, nfft);
        
    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs
    
    return STFT, F, N

def istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0):
    """
    data = istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0)
    
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    
    Inputs:
        X                     : STFT of the signal, to be "inverted"
        window=sinebell(2048) : synthesis window
                                (should be the "complementary" window
                                for the analysis window)
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation
                                (the user has to provide an even number)
                                
    Outputs:
        data                  : time series corresponding to the given
                                STFT the first half-window is removed,
                                complying with the STFT computation
                                given in the function 'stft'
    """
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = np.array(X.shape)
    lengthData = hopsize * (numberFrames - 1) + lengthWindow
    
    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], nfft)
        frameTMP = frameTMP[:lengthWindow]
        data[beginFrame:endFrame] = data[beginFrame:endFrame] \
                                    + window * frameTMP
        
    # remove the extra bit before data that was - supposedly - added
    # in the stft computation:
    data = data[(lengthWindow / 2.0):] 
    return data

# DEFINING THE FUNCTIONS TO CREATE THE 'BASIS' WF0

def generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, stepNotes=4, \
                         lengthWindow=2048, Ot=0.5, perF0=2, \
                         depthChirpInSemiTone=0.5, loadWF0=True,
                         analysisWindow='hanning'):
    """
    F0Table, WF0 = generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048,
                                        stepNotes=4, lengthWindow=2048,
                                        Ot=0.5, perF0=2,
                                        depthChirpInSemiTone=0.5)
                                        
    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    Inputs:
        minF0                the minimum value for the fundamental
                             frequency (F0)
        maxF0                the maximum value for F0
        Fs                   the desired sampling rate
        Nfft                 the number of bins to compute the Fourier
                             transform
        stepNotes            the number of F0 per semitone
        lengthWindow         the size of the window for the Fourier
                             transform
        Ot                   the glottal opening coefficient for
                             KLGLOTT88
        perF0                the number of chirps considered per F0
                             value
        depthChirpInSemiTone the maximum value, in semitone, of the
                             allowed chirp per F0
                             
    Outputs:
        F0Table the vector containing the values of the fundamental
                frequencies in Hertz (Hz) corresponding to the
                harmonic combs in WF0, i.e. the columns of WF0
        WF0     the basis matrix, where each column is a harmonic comb
                generated by KLGLOTT88 (with a sinusoidal model, then
                transformed into the spectral domain)
    """
    # generating a filename to keep data:
    filename = str('').join(['wf0_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_Fs-', str(Fs),
                             '_Nfft-', str(Nfft),
                             '_stepNotes-', str(stepNotes),
                             '_Ot-', str(Ot),
                             '_perF0-', str(perF0),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '.npz'])
    
    if os.path.isfile(filename) and loadWF0:
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0']
    
    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)
    
    # computing the F0 table:
    numberOfF0 = np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))
    
    numberElementsInWF0 = numberOfF0 * perF0
    
    # computing the desired WF0 matrix
    WF0 = np.zeros([Nfft, numberElementsInWF0],dtype=np.double)
    for fundamentalFrequency in np.arange(numberOfF0):
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0, \
                                 analysisWindowType=analysisWindow)
        ##odgd /= np.abs(odgd).max()
        ##odgdSpec = np.fft.fft(np.real(odgd)*np.hanning(lengthWindow), Nfft)
        WF0[:,fundamentalFrequency * perF0] = np.abs(odgdSpec) ** 2
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            print "making some chirped elements..."
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2 
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(odgdSpec) ** 2
            
    np.savez(filename, F0Table=F0Table, WF0=WF0)
    
    return F0Table, WF0

def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='hanning'):
    """
    generateODGDspec:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)
    
    odgd /= np.abs(odgd).max() # added so that less noise after in estimation
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def generate_ODGD_spec_chirped(F1, F2, Fs, lengthOdgd=2048, Nfft=2048, \
                               Ot=0.5, t0=0.0, \
                               analysisWindowType='hanning'):
    """
    generateODGDspecChirped:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F1 = np.double(F1)
    F2 = np.double(F2)
    F0 = np.double(F1 + F2) / 2.0
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType == 'sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType == 'hanning' or \
               analysisWindowType == 'hann':
            analysisWindow = hann(lengthOdgd)
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / np.max(F1, F2))
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 * \
                 (np.exp(-temp_array) \
                  + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                  - (6 * (1 - np.exp(-temp_array)) \
                     / (temp_array ** 2))) \
                  / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(2.0 * 1j * np.pi \
                  * (np.outer(F1 * frequency_numbers,timeStamps) \
                     + np.outer((F2 - F1) \
                                * frequency_numbers,timeStamps ** 2) \
                     / (2 * lengthOdgd / Fs))) \
                     * np.outer(amplitudes,np.ones(lengthOdgd))
    odgd = np.sum(odgd,axis=0)
    
    odgd /= np.abs(odgd).max() # added so that less noise after in estimation
    
    # spectrum:
    odgdSpectrum = np.fft.fft(real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def generate_ODGD(F0, Fs, lengthOdgd=2048, Ot=0.5, \
                  t0=0.0, analysisWindowType='hanning'):
    """
    generate_ODGD:
    
    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """
    
    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)
    
    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)
    
    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)
    
    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    
    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot
    
    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array
    
    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    
    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)
    
    #odgd /= np.abs(odgd).max() # added so that less noise after in estimation
    
    return 2*np.real(odgd), timeStamps[-1]*F0-np.rint(timeStamps[-1]*F0)


def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = np.ceil(numberFrequencyBins \
                                   / ((1.0 - overlap) \
                                      * (numberOfBasis - 1) + 1 \
                                      - 2.0 * overlap))
        # even window length, for convenience:
        lengthSineWindow = 2.0 * np.floor(lengthSineWindow / 2.0) 
        
        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins) 
        
        # size of the "big" window
        sizeBigWindow = 2.0 * numberFrequencyBins
        
        # centers for each window
        ## the first window is centered at, in number of window:
        firstWindowCenter = -numberOfWindowsForUnit + 1
        ## and the last is at
        lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1
        ## center positions in number of frequency bins
        sineCenters = np.round(\
            np.arange(firstWindowCenter, lastWindowCenter) \
            * (1 - overlap) * np.double(lengthSineWindow) \
            + lengthSineWindow / 2.0)
        
        # For future purpose: to use different frequency scales
        isScaleRecognized = True
        
    # For frequency scale in logarithm (such as ERB scales) 
    if frequencyScale == 'log':
        isScaleRecognized = False
        
    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print "The desired feature for frequencyScale is not recognized yet..."
        return 0
    
    # the shape of one window:
    prototypeSineWindow = hann(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[(sizeBigWindow - lengthSineWindow / 2.0):\
              (sizeBigWindow + lengthSineWindow / 2.0)] \
              = np.vstack(prototypeSineWindow)
    
    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])
    
    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[np.int32(mappingFrequency \
                                                    - sineCenters[p] \
                                                    + sizeBigWindow)])
        
    return WGAMMA

# added 17/08/2010 - AR estimation, with basis of generated spectra
# see:
# 
# @conference{moal1997estimation,
#   title={{Estimation de l'ordre et identification des param{\`e}tres d'un processus ARMA}},
#   author={MOAL, N. and FUCHS, J.J.},
#   booktitle={16 Colloque sur le traitement du signal et des images, FRA, 1997},
#   year={1997},
#   organization={GRETSI, Groupe d'Etudes du Traitement du Signal et des Images}
# }
def genARfunction(numberFrequencyBins, sizeOfFourier, Fs, \
                  formantsRange=None, \
                  bwRange=None, \
                  numberOfAmpsPerPole=5, \
                  numberOfFreqPerPole=60, \
                  maxF0 = 1000.0):
    if formantsRange is None:
        formantsRange = [80.0, 1400.0]
    
    if bwRange is None:
        bwMin = maxF0
        bwMax = np.maximum(0.1 * Fs, bwMin)
        bwRange = np.arange(numberOfAmpsPerPole, dtype=np.double) \
                   * (bwMax - bwMin) / \
                   np.double(numberOfAmpsPerPole-1.0) + \
                   bwMin
    
    freqRanges = np.arange(numberOfFreqPerPole) \
                 * (formantsRange[1] - formantsRange[0]) / \
                 np.double(numberOfFreqPerPole-1.0) + \
                 formantsRange[0]
    
    totNbElements = numberOfFreqPerPole * \
                    numberOfAmpsPerPole
    poleAmp = np.zeros(totNbElements)
    poleFrq = np.zeros(totNbElements)
    WGAMMA = np.zeros([numberFrequencyBins, totNbElements])
    cplxExp = np.exp(-1j * 2.0 * np.pi * \
                     np.arange(numberFrequencyBins) / \
                     np.double(sizeOfFourier))
    
    for w in range(numberOfFreqPerPole):
        for a in range(numberOfAmpsPerPole):
            elementNb = w * numberOfAmpsPerPole + a
            poleAmp[elementNb] = np.exp(-bwRange[a] / np.double(Fs))
            poleFrq[elementNb] = freqRanges[w]
            WGAMMA[:,elementNb] = 1 / \
                                  np.abs(1 - \
                                         2.0 * \
                                         poleAmp[elementNb] * \
                                         np.cos(2.0*np.pi*poleFrq[elementNb] / \
                                                np.double(Fs)) * cplxExp +
                                         (poleAmp[elementNb] * cplxExp) ** 2\
                                         ) ** 2
    
    return bwRange, freqRanges, poleAmp, poleFrq, WGAMMA

def genARbasis(numberFrequencyBins, sizeOfFourier, Fs, \
               formantsRange=None, \
               bwRange=None, \
               numberOfAmpsPerPole=5, \
               numberOfFreqPerPole=60, \
               maxF0 = 1000.0):
    if formantsRange is None:
        formantsRange = {}
        formantsRange[0] = [80.0, 1400.0]
        formantsRange[1] = [200.0, 3000.0]
        formantsRange[2] = [300.0, 4000.0]
        formantsRange[3] = [1100.0, 6000.0]
        formantsRange[4] = [4500.0, 15000.0]
        formantsRange[5] = [9000.0, 20000.0]
    
    numberOfFormants = len(formantsRange)
    
    if bwRange is None:
        bwMin = maxF0
        # bwMax = np.maximum(0.1 * Fs, bwMin)
        bwMax = np.maximum(3500, bwMin) # max bandwidth = 4000 Hz. Reasonable
        bwRange = np.arange(numberOfAmpsPerPole, dtype=np.double) \
                   * (bwMax - bwMin) / \
                   np.double(numberOfAmpsPerPole-1.0) + \
                   bwMin
    
    freqRanges = np.zeros([numberOfFormants, numberOfFreqPerPole])
    for n in range(numberOfFormants):
        freqRanges[n] = np.arange(numberOfFreqPerPole) \
                        * (formantsRange[n][1] - formantsRange[n][0]) / \
                        np.double(numberOfFreqPerPole-1.0) + \
                        formantsRange[n][0]
    
    totNbElements = numberOfFreqPerPole * \
                    numberOfFormants * numberOfAmpsPerPole
    poleAmp = np.zeros(totNbElements)
    poleFrq = np.zeros(totNbElements)
    WGAMMA = np.zeros([numberFrequencyBins, totNbElements])
    cplxExp = np.exp(-1j * 2.0 * np.pi * \
                     np.arange(numberFrequencyBins) / \
                     np.double(sizeOfFourier))
    
    for n in range(numberOfFormants):
        for w in range(numberOfFreqPerPole):
            for a in range(numberOfAmpsPerPole):
                elementNb = n * numberOfAmpsPerPole * numberOfFreqPerPole + \
                            w * numberOfAmpsPerPole + \
                            a
                poleAmp[elementNb] = np.exp(-bwRange[a] / np.double(Fs))
                poleFrq[elementNb] = freqRanges[n][w]
                ## pole = poleAmp[elementNb] * \
                ##        np.exp(1j * 2.0 * np.pi * \
                ##               poleFrq[elementNb] / np.double(Fs))
                WGAMMA[:,elementNb] = 1 / \
                   np.abs(1 - \
                          2.0 * \
                          poleAmp[elementNb] * \
                          np.cos(2.0 * np.pi * poleFrq[elementNb] / \
                                 np.double(Fs)) * cplxExp +
                          (poleAmp[elementNb] * cplxExp) ** 2\
                          ) ** 2
    
    return bwRange, freqRanges, poleAmp, poleFrq, WGAMMA

def mel2hz(f):
    return 700.0 * (10**(f / 2595.0) - 1)

def hz2mel(f):
    return 2595 * np.log10(1+f/700.0)

def generate_ODGD_vec(F0vec, Fs, lengthOdgd=2048,
                      Ot=0.5, 
                      t0=0.0, analysisWindowType='hanning',
                      noiseLevel=1.):
    """generate_ODGD_vec
    """
    N = F0vec.size
    odgd = np.zeros([N*lengthOdgd])
    
    for n in range(N):
        time0 = n*lengthOdgd
        time8 = time0 + lengthOdgd
        if F0vec[n]==0:
            odgdtmp = np.random.randn(lengthOdgd)
            t0 = np.random.randn()
        else:
            odgdtmp, t0 = generate_ODGD(F0=np.double(F0vec[n]),
                                        Fs=np.double(Fs),
                                        lengthOdgd=lengthOdgd,
                                        Ot=Ot, 
                                        t0=t0, analysisWindowType='hanning')
            odgdtmp /= np.double(F0vec[n])
        odgd[time0:time8] = odgdtmp
    
    return odgd

def synthesis(fs, steWin, lenWin, lenFT,
              W, # the different dictionary matrices,
                 # no need for estim. amplitudes!
              WR, GR, # smoothed frequency filter
              H, # amplitude component
              mu, # state sequence
              F0Table, # F0 frequencies for W[0]
              chirpPerF0=1):
    """synthesis from the SFSNMF parameters
    """
    myF, myNF0 = W[0].shape
    myNFrames = mu[0].shape
    
    # TODO: maybe there could be issues with mu[0], if
    #    notably mu[0]>F0Table.size...
    F0seq = F0Table[np.int32(mu[0] / chirpPerF0)]
    
    odgd = generate_ODGD_vec(F0vec=F0seq,
                             Fs=fs,
                             lengthOdgd=steWin,
                             Ot=0.5, 
                             t0=0.0,
                             analysisWindowType='hanning',
                             noiseLevel=1.)
    
    ODGD, F, N = stft(data=odgd,
                      window=hann(lenWin),
                      hopsize=hopsize,
                      nfft=lenFT,
                      fs=fs)
    
    # ODGD should be at least same length as obs:
    ODGD = ODGD[:myF]
    # ODGD /= np.vstack(np.abs(ODGD).max(axis=0))
    ODGD /= np.abs(ODGD).max(axis=0)
    ODGD = np.concatenate((ODGD,
                           np.zeros([mu[0].size-ODGD.shape[1],
                                     myF]).T), axis=1)
    # First, filter part: sqrt of product of the rest of parameters:
    # (taking log, for better numerical stability)
    S = np.sum([0.5*np.log(W[n][:,mu[n]]) \
                for n in range(1, len(W))],
               axis=0) +\
               0.5 * np.log(H) + \
               0.5 * np.log(np.vstack(np.dot(WR, GR)))
    S = np.exp(S)
    # to lower unvoiced components:
    S[:,mu[0]==(myNF0-1)] = S[:,mu[0]==(myNF0-1)] / np.sqrt(lenFT)
    #S = ODGD * np.prod([np.sqrt(W[n][:,mu[n]]) \
    #                    for n in range(1, len(W)-1)],
    #                   axis=0)
    #S *= np.sqrt(H * np.vstack(np.dot(WR, GR)))
    
    # Source/Filter resulting matrix:
    S = S * ODGD 
    
    synth = istft(X=S,
                  window=hann(lenWin),
                  hopsize=np.double(steWin),
                  nfft=lenFT)
    
    return synth, S, ODGD, F0seq, odgd

def synthesisUVEn(fs, steWin, lenWin, lenFT,
                  W, # the different dictionary matrices, 
                  WR, GR, # smoothed frequency filter
                  H, # amplitude component
                  GUV, # amplitude of unvoiced components
                  mu, # state sequence
                  F0Table, # F0 frequencies for W[0]
                  chirpPerF0=1):
    """synthesis from the SFSNMF parameters
    
    includes unvoiced component energy as well.
    """
    myF, myNF0 = W[0].shape
    myNFrames = mu[0].shape
    
    # TODO: maybe there could be issues with mu[0], if
    #    notably mu[0]>F0Table.size...
    F0seq = F0Table[np.int32(mu[0] / chirpPerF0)]
    
    odgd = generate_ODGD_vec(F0vec=F0seq,
                             Fs=fs,
                             lengthOdgd=steWin,
                             Ot=0.5, 
                             t0=0.0,
                             analysisWindowType='hanning',
                             noiseLevel=1.)
    
    ODGD, F, N = stft(data=odgd,
                      window=hann(lenWin),
                      hopsize=hopsize,
                      nfft=lenFT,
                      fs=fs)
    
    # ODGD should be at least same length as obs:
    ODGD = ODGD[:myF]
    # ODGD /= np.vstack(np.abs(ODGD).max(axis=0))
    ODGD /= np.abs(ODGD).max(axis=0)
    ODGD = np.concatenate((ODGD,
                           np.zeros([mu[0].size-ODGD.shape[1],
                                     myF]).T), axis=1)
    ODGD += np.sqrt(GUV[0]/2.) * \
            (np.random.randn(myF, mu[0].size) + \
             np.random.randn(myF, mu[0].size) * 1j)
    # First, filter part: sqrt of product of the rest of parameters:
    # (taking log, for better numerical stability)
    S = np.sum([0.5*np.log(W[n][:,mu[n]] + GUV[n]) \
                for n in range(1, len(W))],
               axis=0) +\
               0.5 * np.log(H) + \
               0.5 * np.log(np.vstack(np.dot(WR, GR)))
    S = np.exp(S)
    # to lower unvoiced components:
    S[:,mu[0]==(myNF0-1)] = S[:,mu[0]==(myNF0-1)] / np.sqrt(lenFT)
    #S = ODGD * np.prod([np.sqrt(W[n][:,mu[n]]) \
    #                    for n in range(1, len(W)-1)],
    #                   axis=0)
    #S *= np.sqrt(H * np.vstack(np.dot(WR, GR)))
    
    # Source/Filter resulting matrix:
    S = S * ODGD 
    
    synth = istft(X=S,
                  window=hann(lenWin),
                  hopsize=np.double(steWin),
                  nfft=lenFT)
    
    return synth, S, ODGD, F0seq, odgd
    
def synthesisFull(fs, steWin, lenWin, lenFT,
                  W, G, # the different dictionary matrices,
                        # with estim. amplitudes, this time
                  WR, GR, # smoothed frequency filter
                  H, # amplitude component
                  mu, # state sequence
                  F0Table, # F0 frequencies for W[0]
                  chirpPerF0=1):
    """synthesis from the SFSNMF parameters
    
    full resynthesis using all parameters from the SFSNMF model.
    """
    myF, myNF0 = W[0].shape
    myNFrames = mu[0].shape
    
    # TODO: maybe there could be issues with mu[0], if
    #    notably mu[0]>F0Table.size...
    F0seq = F0Table[np.int32(mu[0] / chirpPerF0)]
    
    odgd = generate_ODGD_vec(F0vec=F0seq,
                             Fs=fs,
                             lengthOdgd=steWin,
                             Ot=0.5, 
                             t0=0.0,
                             analysisWindowType='hanning',
                             noiseLevel=1.)
        
    ODGD, F, N = stft(data=odgd,
                      window=hann(lenWin),
                      hopsize=hopsize,
                      nfft=lenFT,
                      fs=fs)
    
    # ODGD should be at least same length as obs:
    ODGD = ODGD[:myF]
    ODGD /= abs(ODGD).max(axis=0)
    ODGD = np.concatenate((ODGD,
                           np.zeros([mu[0].size-ODGD.shape[1],
                                     myF]).T), axis=1)
    # First, filter part: sqrt of product of the rest of parameters:
    # (taking log, for better numerical stability)
    S = np.sum([0.5*np.log(np.dot(W[n], G[n])) \
                for n in range(1, len(W))],
               axis=0) +\
               0.5 * np.log(H) + \
               0.5 * np.log(np.vstack(np.dot(WR, GR)))
    S = np.exp(S)
    # to lower unvoiced components:
    #S[:,mu[0]==(myNF0-1)] = S[:,mu[0]==(myNF0-1)] * 0.4
    #S = ODGD * np.prod([np.sqrt(W[n][:,mu[n]]) \
    #                    for n in range(1, len(W)-1)],
    #                   axis=0)
    #S *= np.sqrt(H * np.vstack(np.dot(WR, GR)))
    
    # Source/Filter resulting matrix:
    S = S * ODGD 
    
    synth = istft(X=S,
                  window=hann(lenWin),
                  hopsize=np.double(steWin),
                  nfft=lenFT)
    
    return synth, S, ODGD, F0seq, odgd

# Running the tests of above functions:

displayEvolution = True
class Options(object):
    def __init__(self, verbose=True):
        self.verbose = verbose

options = Options()

import matplotlib.pyplot as plt
from imageMatlab import imageM

## plt.rc('text', usetex=True)
plt.rc('image',cmap='jet')
plt.rc('lines',markersize=4.0)
plt.ion()

# TODO: also process these as options:
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

import os
if os.path.isdir('/Users/jeanlouis/work/BDD/'):
    pathToArx = '/Users/jeanlouis/work/BDD/formants/'
    annotPath = '/Users/jeanlouis/work/BDD/formants/VTRFormants/'
    prefixBDD = '/Users/jeanlouis/work/BDD/formants/VTRFormants/'
    audioPath = '/users/jeanlouis/work/BDD/formants/timit/db/timit/'
    pathToYin = '/users/jeanlouis/work/svn/speedlingua/trunk/yin/'
    pathToHMM = '/users/jeanlouis/work/svn/PersoDurrieu/'+\
                'programmation/python/hmm/'
elif os.path.isdir('/home/durrieu/work/BDD/'):
    pathToArx = '/home/durrieu/work/BDD/formants/'
    annotPath = '/home/durrieu/work/BDD/formants/VTRFormants/'
    prefixBDD = '/home/durrieu/work/BDD/formants/VTRFormants/'
    audioPath = '/home/durrieu/work/BDD/formants/timit/db/timit/'
    pathToYin = '/home/durrieu/work/svnepflch/speedlingua/trunk/yin/'
    pathToHMM = '/home/durrieu/work/svnEPFL//'+\
                'programmation/python/hmm/'

files = mt.recursiveSearchFromRoot(annotPath, conditionExtension='.fb')
files.sort()
nbFilesHill = len(files)

numberGTFormants = 4

windowSizeInSeconds = 0.064 # for hill: 0.032
fs = 16000
Fs=fs
lenWin = nextpow2(np.ceil(fs * windowSizeInSeconds))
steWin = int(10. / 1000. *fs)# lenWin/4 # in accordance with annotation data
hopsize = steWin#windowSizeInSamples/4
NFT = lenWin
maxFinFT = 8000
F = np.ceil(maxFinFT * NFT / np.double(fs))

comments = 'keeping parameters for resynthesis purpose, '+\
           'setting same formant ranges as for va-sffhmm'
chirpPerF0 = 1
ninner = 3
dispMat = True
verbose = True

displayStatic = True#False #True
displayEvolution =  False #False#False

Ot = .6
numberOfAmpsPerPole = 5
niterSpSm = 30
numberGTFormants = 4

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

bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = \
    genARbasis(F, NFT, Fs, maxF0=maxF0,
               formantsRange=formantsRange,
               numberOfAmpsPerPole=numberOfAmpsPerPole)
numberOfFormants = freqRanges.shape[0]
numberOfAmpPerFormantFreq = bwRange.size

poleFrqMel = hz2mel(poleFrq)

WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
Fwgamma, Nwgamma = WGAMMA.shape

nbElPerF = Nwgamma/numberOfFormants

F0Table, WF0 = \
    generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, \
                         stepNotes=stepNotes, \
                         lengthWindow=lenWin, Ot=Ot, \
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
    F0Table = np.concatenate((F0Table, [0]))

## W = [WF0.copy()]
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
                       sizeOfFourier=NFT, Fs=Fs,
                       frequencyScale='linear',
                       numberOfBasis=numberOfBasisWR, 
                       overlap=.75)

P = len(W)

import datetime
currentTime = datetime.datetime.strftime(\
    datetime.datetime.now(), format='%Y%m%dT%H%M')

dirSaveArx = str('').join([prefixBDD,
                           '/result_Ot-', str(Ot),
                           '_nbFP-', str(numberOfFormants),
                           '_numberOfBasisWR-', str(numberOfBasisWR),
                           '_winSize-',str(lenWin),
                           '_hopsize-',str(hopsize),
                           '_niterSpSm-',str(niterSpSm),
                           '_', currentTime,
                           '/'])

if not(os.path.isdir(dirSaveArx)):
    os.mkdir(dirSaveArx)

fileSaveParam = str('').join([dirSaveArx,
                              '/commonParameters.npz'])
np.savez(fileSaveParam, Ot=Ot, W=W, F0Table=F0Table, WGAMMA=WGAMMA,
         poleFrq=poleFrq, poleAmp=poleAmp, 
         formantsRange=formantsRange,
         numberOfFormants=numberOfFormants)

savefilename = str('').join(['sfsnmf_result_Ot-', str(Ot),
                             '_nbFP-', str(numberOfFormants),
                             '_numberOfBasisWR-', str(numberOfBasisWR),
                             '_winSize-',str(lenWin),
                             '_hopsize-',str(steWin),
                             '_niterSpSm-',str(niterSpSm),
                             '_', currentTime,
                             '.npz'])

phonesAll = ['h#', 'q', 'ih', 'n', 'w', 'ey', 'dcl', 'jh', 'ix', 'gcl', \
          'g', 'ow', 'sh', 'iy', 's', 'epi', 'dh', 'd', 'tcl', 't', 'r', \
          'bcl', 'b', 'aa', 'z', 'eh', 'y', 'ux', 'nx', 'ng', 'el', 'hh', \
          'v', 'ao', 'pcl', 'p', 'pau', 'kcl', 'k', 'ah', 'm', 'l', 'axr', \
          'ae', 'dx', 'ay', 'f', 'ax', 'oy', 'uh', 'er', 'ax-h', 'ch', \
          'hv', 'th', 'en', 'aw', 'uw', 'eng', 'zh', 'em']

nbPhon = len(phonesAll)

erFilMatrix = {}
erAbsMatrix = {}
erAbsMatFrq = {}
for n in range(nbFilesHill):
    erFilMatrix[n] = {}
    erAbsMatrix[n] = {}
    erAbsMatFrq[n] = {}

nip = 0

params = {'windowSizeInSeconds':windowSizeInSeconds,
          'fs':fs,
          'lenWin':lenWin,
          'steWin':steWin,
          'NFT':NFT, 'maxFinFT':maxFinFT,
          'F':F, 'chirpPerF0':chirpPerF0,
          'ninner': niterSpSm,
          'withExtraUnvoiced': withExtraUnvoiced,
          'withFlatFilter': True, # done in hard in line 726 
          'currentTime':currentTime,
          'comments':comments,
          'chirpPerF0': chirpPerF0}

#for filenb in range((nip*nbFilesHill/4),((nip+1)*nbFilesHill/4)):
for filenb, vtrfile in enumerate(files):
# for filenb, vtrfile in enumerate([files[0]]):
    print filenb, nbFilesHill, vtrfile
    vtrfile = files[filenb]
    suffix = vtrfile.split(annotPath)[-1].lower()[:-2]
    wavfile = audioPath+suffix+'wav'
    fs, data, metadata = st.readNISTwav(wavfile)
    data = data[0]
    if metadata['channel_count']!=1:
        warnings.warn("Channel number is "+\
                      "%d, we keep only first one."%metadata['channel_count'])
    data = np.double(data)
    data /= (abs(data)).max()
    
    X, freqs, timeStamps = stft(data, fs=fs, hopsize=hopsize,
                                window=np.hanning(lenWin), nfft=NFT)
    
    SX = np.abs(X) ** 2
    SX = SX[:F,:]
    
    nframe, samPeriod, sampSize, numComps, fileType, trueFormants = \
        st.readFormantMS(vtrfile)
    trueFormants = trueFormants[:,:4].T # kHz
    
    print "Running SFSNMF:"
    G, GR, H, recoError1 = ARPIMM.SFSNMF(SX, W, WR,
                                         poleFrq=None, H0=None, \
                                         stepNotes=None,
                                         nbIterations=niterSpSm,
                                         verbose=verbose,
                                         dispEvol=displayEvolution)
    
    # draw figure plus formant annotated
    FHzmax = 4000.0
    nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
    ytickslab = np.array([1000, 2000, 3000, 4000])
    ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
    fontsize = 16
    figheight=4.5
    figwidth=9.0
    
    # Compute the estimated formants and estimated spectra
    eps = 10**-50
    mu = {}
    #mu[0] = np.dot(np.arange(NF0-1) * \
    #               (np.arange(NF0-1, 0, -1))**2, \
    #               G[0][:-1,:]) / \
    #               np.dot((np.arange(NF0-1, 0, -1))**2,
    #                      np.maximum(G[0][:-1,:], eps))
    mu[0] = G[0].argmax(axis=0)
    # mu[0][G[0][-1,:]>G[0][:-1,:].sum(axis=0)] = NF0 - 1
    Shat = [np.dot(W[0], G[0])]
    #Shat = [W[0][:,np.int16(np.rint(mu[0]))]]
    GUV = [G[0][-1]]
    
    for p in range(1,P):
        #mu[p] = np.dot(np.arange(nbElPerF) * \
        #               (np.arange(nbElPerF, 0, -1))**2,
        #               G[p][:-1,:]) / \
        #               np.dot((np.arange(nbElPerF, 0, -1))**2,
        #                      np.maximum(G[p][:-1,:], eps))
        mu[p] = G[p].argmax(axis=0)
        # mu[p][G[p][-1,:]>G[p][:-1,:].sum(axis=0)] = nbElPerF - 1
        Shat.append(np.dot(W[p],G[p]))
        #Shat.append(W[p][:,np.int16(np.rint(mu[p]))])
        GUV.append(G[p][-1])
    
    F0seq = F0Table[np.int32(mu[0])]
    
    SXhat = np.log(H) + \
            np.vstack(np.log(np.dot(WR,GR))) + \
            np.sum(np.log(Shat), axis=0)
    SXhat = np.exp(SXhat)
    
    Fformant = {}
    for p in range(1,P):
        Fformant[p-1] = poleFrq[np.int32((p-1)*nbElPerF + \
                                         np.minimum(mu[p],
                                                    nbElPerF-1))]
        Fformant[p-1][mu[p]==nbElPerF] = 0
    
    statesInMel = np.zeros([numberOfFormants, H.size])
    for p in range(1, numberOfFormants+1):
        nbElPerF = G[p].shape[0]-1 
        statesInMel[p-1] = poleFrqMel[\
            np.int32((p-1)*nbElPerF+\
                     np.minimum(mu[p],
                                nbElPerF-1))]
        statesInMel[p-1][mu[p]==nbElPerF] = 0
        
    
    # computing the error scores:
    txt = open(vtrfile[:-2]+'phn')
    phones = txt.readlines()
    txt.close()
    for phnb, ph in enumerate(phones):
        ph_ = ph.strip('\n')
        elts = ph_.split(' ')
        # start and stop of phone, in frames
        start = int(elts[0]) / steWin#/np.double(fs)
        stop = int(elts[1]) / steWin#/np.double(fs)
        stop = min(stop, SX.shape[1])
        phon = elts[2]
        #print start, stop, phon, nframe
        erFilMatrix[filenb][str(phnb)+'_'+phon] = \
            np.ones([numberGTFormants,
                     numberOfFormants])
        erAbsMatrix[filenb][str(phnb)+'_'+phon] = \
            np.ones([numberGTFormants,
                     numberOfFormants])
        erAbsMatFrq[filenb][str(phnb)+'_'+phon] = \
            np.ones([numberGTFormants,
                     numberOfFormants])
        stop = min(stop, nframe)
        for n in range(numberGTFormants):
            erFilMatrix[filenb][str(phnb)+'_'+phon][n] = \
                np.sqrt(np.mean((statesInMel[:,start:stop] - \
                    hz2mel(trueFormants[n, start:stop]*1000.))**2, axis=1))
            erAbsMatrix[filenb][str(phnb)+'_'+phon][n] = \
                np.mean(np.abs(statesInMel[:,start:stop] - \
                    hz2mel(trueFormants[n, start:stop]*1000.)), axis=1)
            erAbsMatFrq[filenb][str(phnb)+'_'+phon][n] = \
                np.mean(np.abs(mel2hz(statesInMel[:,start:stop]) - \
                    (trueFormants[n, start:stop]*1000.)), axis=1)
    
    print erFilMatrix[filenb]
    
    #fileSaveArx = str('').join([dirSaveArx,
    #                            '/', suffix,
    #                            'npz'])
    
    #np.savez(fileSaveArx, trueFormants=trueFormants,
    #         Fformant=Fformant,
    #         H=H, GR=GR, G=G,
    #         recoError1=recoError1,
    #         F0seq=F0seq, 
    #         filenb=filenb, mu=mu)
    
    # drawing:
    if dispMat:
        fig = plt.figure(10)
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax1.imshow(np.log(np.abs(SX)), interpolation='nearest')
        ax1.set_title(str(filenb)+'_'+wavfile.split('/')[-1] + ': observation')
        for n in range(numberGTFormants): # Ground Truth
            ax1.plot(trueFormants[n]*1000./np.double(fs)*NFT,
                     'ok')
        ##ax1.plot(F0seq/ \
        ##         np.double(fs) * NFT, '.-')
        ##for n in range(numberOfFormants):
        ##    ax1.plot(mel2hz(statesInMel[n])/ \
        ##             np.double(fs) * NFT, '.-')
        #.colorbar()
        ax1.axis('tight')
        ax2=fig.add_subplot(212, sharex=ax1,sharey=ax1)
        ax2.imshow(np.log(np.abs(SXhat)), interpolation='nearest')
        ax2.set_title('posterior mean')
        ##for n in range(numberGTFormants): # Ground Truth
        ##    ax2.plot(trueFormants[n]*1000./np.double(fs)*NFT,
        ##             'ok')
        ax2.plot(F0seq/ \
                 np.double(fs) * NFT, '.')
        for n in range(numberOfFormants):
            ax2.plot(mel2hz(statesInMel[n])/ \
                     np.double(fs) * NFT, '.')
        # plt.colorbar()
        #img1 = ax1.get_images()[0]
        #img2 = ax2.get_images()[0]
        #img1.set_colorbar(img1, ax1)
        #img2.set_colorbar(img2, ax2)
        ax2.get_images()[0].set_clim(ax1.get_images()[0].get_clim())
        xticks2 = ax2.get_xticks()[1:-1]
        ax2.set_xticks(xticks2)
        ax2.set_xticklabels(xticks2*steWin/(1.*fs))
        yticklabels2 = np.arange(start=1,stop=fs/2000,step=2)
        yticks2 = yticklabels2 * (1.*NFT) / fs * 1000
        ax2.set_yticks(yticks2)
        ax2.set_yticklabels(yticklabels2)
        ax2.set_xlabel('Time (s)')
        ax1.set_ylabel('Freq. (kHz)')
        ax2.set_ylabel('Freq. (kHz)')
        ax2.axis('tight')
        fig.canvas.draw()
        figname = audioPath+suffix+'_SFSNMF_'+currentTime+'.pdf'
        plt.savefig(figname)
    
    # reestimate parameters, only on formant path:
    # This second estimation round is meant to improve the resynthesis
    # feature, by estimating the parameters only on the estimated formant
    # tracks.
    G0 = dict()
    for p in range(len(G)):
        G0[p] = np.zeros(G[p].shape)
        G0[p][(np.int16(mu[p]),range(mu[p].size))] = \
            G[p][(np.int16(mu[p]),range(mu[p].size))]
        G0[p][-1] = G[p][-1]
    
    niterSpSm2 = 4
    G2, GR2, H2, recoError2 = ARPIMM.SFSNMF(SX, W, WR,
                                            G0=G0,
                                            poleFrq=None, H0=H, \
                                            stepNotes=None,
                                            nbIterations=niterSpSm2,
                                            verbose=verbose,
                                            dispEvol=displayEvolution,
                                            smoothIt=False)
    
    # Compute the estimated formants and estimated spectra
    eps = 10**-50
    mu2 = {}
    mu2[0] = G2[0].argmax(axis=0)
    GUV = [G2[0][-1]]
    Shat2 = [W[0][:,np.int16(np.rint(mu2[0]))] +\
             np.outer(W[0][:,-1], G2[0][-1])] # hard constraint estimate
    Shat2_ = [np.dot(W[0], G2[0])] # soft estimate, but not data compressed
    for p in range(1, P):
        mu2[p] = G2[p].argmax(axis=0)
        Shat2_.append(np.dot(W[p],G2[p]))
        Shat2.append(W[p][:,np.int16(np.rint(mu2[p]))] +\
                     np.outer(W[p][:,-1], G2[p][-1]))
        GUV.append(G2[p][-1])
    
    F0seq2 = F0Table[np.int32(mu2[0])]
    
    SXhat2 = np.log(H2) + \
             np.vstack(np.log(np.dot(WR,GR2))) + \
             np.sum(np.log(Shat2), axis=0)
    SXhat2 = np.exp(SXhat2)
    SXhat2_ = np.log(H2) + \
              np.vstack(np.log(np.dot(WR,GR2))) + \
              np.sum(np.log(Shat2_), axis=0)
    SXhat2_ = np.exp(SXhat2_)
    # drawing:
    if dispMat:
        fig = plt.figure(11)
        fig.clf()
        ax1 = fig.add_subplot(211)
        # ax1.imshow(np.log(np.abs(SXhat2)), interpolation='nearest')
        ax1.imshow(np.log(np.abs(SX)), interpolation='nearest')
        ax1.get_images()[0].set_clim([np.log(SX).min(),
                                      np.log(SX).max()])
        ax1.set_title('Original spectrum')
        for n in range(numberGTFormants): # Ground Truth
            ax1.plot(trueFormants[n]*1000./np.double(fs)*NFT,
                     'ok')
        ax1.plot(F0seq/ \
                 np.double(fs) * NFT, '.')
        for n in range(numberOfFormants):
            ax1.plot(mel2hz(statesInMel[n])/ \
                     np.double(fs) * NFT, '.')
        ax1.axis('tight')
        ax2 = fig.add_subplot(212, sharex=ax1,sharey=ax1)
        ax2.imshow(np.log(np.abs(SXhat2_)), interpolation='nearest')
        ax2.set_title('Estimated spectrum (second estimate)')
        ax2.plot(F0seq2/ \
                 np.double(fs) * NFT, '.')
        for n in range(numberOfFormants):
            ax2.plot(mel2hz(statesInMel[n])/ \
                     np.double(fs) * NFT, '.')
        ax2.get_images()[0].set_clim(ax1.get_images()[0].get_clim())
        xticks2 = ax2.get_xticks()[1:-1]
        ax2.set_xticks(xticks2)
        ax2.set_xticklabels(xticks2*steWin/(1.*fs))
        yticklabels2 = np.arange(start=1,stop=fs/2000,step=2)
        yticks2 = yticklabels2 * (1.*NFT) / fs * 1000
        ax2.set_yticks(yticks2)
        ax2.set_yticklabels(yticklabels2)
        ax2.set_xlabel('Time (s)')
        ax1.set_ylabel('Freq. (kHz)')
        ax2.set_ylabel('Freq. (kHz)')
        ax2.axis('tight')
        fig.canvas.draw()
        figname = audioPath+suffix+'_SFSNMF_'+currentTime+'.pdf'
        plt.savefig(figname)
        figname = audioPath+suffix+'_SFSNMF_'+currentTime+'.png'
        plt.savefig(figname)
    
    np.savez(savefilename,
             erAbsMatrix=erAbsMatrix,
             erFilMatrix=erFilMatrix,
             erAbsMatFrq=erAbsMatFrq,
             params=params)
    
    oggfilename = wavfile[:-3]+'ogg'
    oggsyntname = wavfile[:-4]+'_SFSNMF_synth.ogg'
    al.oggwrite(data=data,filename=oggfilename,fs=fs)
    # H3 = H2 +..
    # issue: this does not work, for some reason.
    # seems that some energy is missing...
    #datas = synthesis(fs=fs, steWin=steWin, lenWin=lenWin, lenFT=NFT,
    #                  W=W, # the different dictionary matrices, 
    #                  WR=WR, GR=GR2, # smoothed frequency filter
    #                  H=H2, # amplitude component
    #                  mu=[np.int16(np.rint(mu2[n]))\
    #                      for n in mu2], # state sequence
    #                  F0Table=F0Table, # F0 frequencies for W[0]
    #                  chirpPerF0=chirpPerF0)
    datas = synthesisUVEn(fs=fs, steWin=steWin, lenWin=lenWin, lenFT=NFT,
                          W=W, # the different dictionary matrices, 
                          WR=WR, GR=GR2, # smoothed frequency filter
                          H=H2, # amplitude component
                          GUV=[np.float16(GUV[n])\
                               for n in range(len(GUV))],
                          # amplitude of unvoiced components
                          mu=[np.int16(np.rint(mu2[n]))\
                              for n in mu2], # state sequence
                          F0Table=F0Table, # F0 frequencies for W[0]
                          chirpPerF0=chirpPerF0)
    # 20120515: OK, the following seems to work
    # testing resynth without the unvoiced elt in W[0]:
    #W3 = dict(W)
    ##W3[0] = W3[0][:,-3:-1]#[:,:-1]
    #G3 = dict(G2)
    ##G3[0] = G3[0][-3:-1]#[:-1]
    #datas = synthesisFull(fs=fs, steWin=steWin, lenWin=lenWin, lenFT=NFT,
    #                  W=W3, G=G3,# the different dictionary matrices, 
    #                  WR=WR, GR=GR2, # smoothed frequency filter
    #                  H=H2, # amplitude component
    #                  mu=[np.int16(np.rint(mu2[n]))\
    #                      for n in mu2], # state sequence
    #                  F0Table=F0Table, # F0 frequencies for W[0]
    #                  chirpPerF0=chirpPerF0)
    datasyn = datas[0] / (abs(datas[0]).max()+.1)
    #datasyn = datas[4] / (abs(datas[4]).max()+.1)
    al.oggwrite(data=datasyn,filename=oggsyntname,fs=fs)
    datasynthParams = wavfile[:-4]+'_SFSNMF_'+currentTime+'.npz'
    np.savez(datasynthParams, H=H2, GR=GR2,
             mu=[np.int16(np.rint(mu2[n]))\
                 for n in mu2],
             steWin=steWin, lenWin=lenWin, NFT=NFT, fs=fs,
             GUV=[np.float16(GUV[n])\
                  for n in range(len(GUV))])

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
    plt.xlabel('Frequency (Hz)', fontsize=fontsize)
    plt.legend((['F0 = %dHz' %(F0Table[nbF0])]))
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
    
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
    plt.xlabel('Frequency (Hz)', fontsize=fontsize)
    plt.axis('tight')
    plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
