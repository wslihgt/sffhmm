#!/usr/bin/python

import numpy as np
import os.path
#import SIMM
import time
import ARPIMM
reload(ARPIMM)

import scipy.linalg as spla

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
                                         np.cos(2.0 * np.pi * poleFrq[elementNb] / \
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
        bwMax = np.maximum(0.1 * Fs, bwMin)
        # bwMax = np.maximum(3500, bwMin) # max bandwidth = 4000 Hz. Reasonable?
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

### according to hillenbrand American vowels 1995
##formantsRange = {}
##formantsRange[0] = [340.0, 1000.0]
##formantsRange[1] = [900.0, 3100.0]
##formantsRange[2] = [1700.0, 3800.0]
##formantsRange[3] = [3300.0, 4700.0] 
##formantsRange[4] = [4500.0, np.minimum(15000.0, maxFinFT)] # adding one for full band

# according to schafer/rabiner 1969 # good for m01...
##formantsRange = {}
##formantsRange[0] = [200.0, 900.0]
##formantsRange[1] = [550.0, 2700.0]
##formantsRange[2] = [1100.0, 2950.0]
##formantsRange[3] = [2400.0, 6000.0] # adding one for full band
##formantsRange[4] = [4500.0, np.minimum(15000.0, maxFinFT)] # adding one for full band

# schafer + hill (union), 20101019
##formantsRange = {}
##formantsRange[0] = [200.0, 1000.0]
##formantsRange[1] = [550.0, 3100.0]
##formantsRange[2] = [1700.0, 3800.0]
##formantsRange[3] = [2400.0, 6000.0] # adding one for full band
##formantsRange[4] = [4500.0, np.minimum(15000.0, maxFinFT)] # adding one for full band

if True : # get the data from hillenbrand
    from loadHillenbrand import *
    
    fs=16000.0
    Fs=fs
    
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
    #formantsRange[4] = [3300.0, 7000.0]
    formantsRange[5] = [4500.0, 8000.0]
    formantsRange[6] = [5500.0, 8000.0]
    # formantsRange[7] = [6500.0, 8000.0]
    
    numberOfFormants = len(formantsRange)
    
    bwRange, freqRanges, poleAmp, poleFrq, WGAMMA = \
             genARbasis(F, NFT, Fs, maxF0=maxF0,
                        formantsRange=formantsRange,
                        numberOfAmpsPerPole=numberOfAmpsPerPole,
                        numberOfFreqPerPole=numberOfFreqPerPole)
    numberOfFormants = freqRanges.shape[0]
    numberOfAmpPerFormantFreq = bwRange.size
    
    poleFrqMel = hz2mel(poleFrq)
    
    WGAMMA = WGAMMA / np.outer(np.ones(WGAMMA.shape[0]), WGAMMA.max(axis=0))
    Fwgamma, Nwgamma = WGAMMA.shape
    
    nbElPerF = Nwgamma/numberOfFormants
    
    F0Table, WF0 = \
             generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, \
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
                                  '/commonParameters.npz'])
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
    
    #snrs = (
    #    -10., -5., 0., 5., 10., 20., 30)
    snrs = (
        20., )
    
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
    
    nbFilesHill = timedat.shape[0]
    ## mpl.mlab.find(bigdatnames=='m07oa')[0] # 1163
    ## ae : 100
    ## oa : 1163
    
    nip = 3
    
    for filenb in range(nbFilesHill):# range((nip*nbFilesHill/4),((nip+1)*nbFilesHill/4)):
        ##filenb = 100 # for bigdat
        if True or bigdatnames[filenb][:3] == 'w01':
            filenbt = mpl.mlab.find(timedatnames==bigdatnames[filenb])[0]
            print "processing file %s, number %d of %d" %(bigdatnames[filenb],
                                                          filenb,
                                                          nbFilesHill)
            
            for snrnb, snr in enumerate(snrs):
                fs, data = loadHill(bigdatnames[filenb])
                data = data / \
                       2.0**(data[0].nbytes * \
                             8.0 - 1) # -1 because signed int.
                
                # adding noise:
                print 'snr', snr, 'effective', 10*np.log10(
                    (data**2).sum()
                    /(1.*(np.sqrt(data.var() * (10 ** (- snr / 10.)))
                          * np.random.randn(data.size))**2).sum()
                    )
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
                nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
                ytickslab = np.array([1000, 2000, 3000, 4000])
                ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
                fontsize = 16
                figheight=4.5
                figwidth=9.0
                
                displayStatic = False #True #
                displayEvolution = False#False #True#
                
                print "Running SFSNMF:"
                G, GR, H, recoError1 = ARPIMM.SFSNMF(SX, W, WR,
                                                     poleFrq=None, H0=None, \
                                                     stepNotes=None,
                                                     nbIterations=niterSpSm,
                                                     verbose=False,
                                                     dispEvol=displayEvolution)
                
                if displayStatic:
                    plt.figure(3)#, figsize=[figwidth, figheight])
                    plt.clf()
                    plt.title(bigdatnames[filenb])
                    plt.imshow(db(SX))
                    clmt = plt.figure(3).get_axes()[0].get_images()[0].get_clim()
                    plt.hold(True)
                    
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
                Shat = [np.dot(W[0], G[0])]
                mu = {}
                mu[0] = np.dot(np.arange(NF0-1) * \
                               (np.arange(NF0-1, 0, -1))**2, \
                               G[0][:-1,:]) / \
                               np.dot((np.arange(NF0-1, 0, -1))**2,
                                      np.maximum(G[0][:-1,:], eps))
                
                ##plt.figure()
                ##plt.subplot(P,1,1)
                ##imageM(db(Shat[0]))
                for p in range(1,P):
                    mu[p] = np.dot(np.arange(nbElPerF) * \
                                   (np.arange(nbElPerF, 0, -1))**2,
                                   G[p][:-1,:]) / \
                                   np.dot((np.arange(nbElPerF, 0, -1))**2,
                                          np.maximum(G[p][:-1,:], eps))
                    ##plt.subplot(P,1,p+1)
                    ##plt.figure()
                    Shat.append(np.dot(W[p],G[p]))
                    ##imageM(db(Shat[p]))
                    
                F0seq = F0Table[np.int32(mu[0])]
                
                compTimes[snrnb + filenb*len(snrs)] = time.time() - time0
                
                SXhat = H * np.vstack(np.dot(WR,GR)) * np.prod(Shat, axis=0)
                
                if displayStatic:
                    plt.figure(4)
                    plt.clf()
                    plt.title(bigdatnames[filenb])
                    ##imageM(db(np.prod(Shat, axis=0)))
                    plt.imshow(db(SXhat))#, cmap=mpl.cm.gray_r)
                    plt.clim(clmt)
                    for n in range(3):
                        plt.plot(timeStampsInFrame,
                                 bigdat[filenb,5+n+3*np.arange(8)]/\
                                 np.double(fs)*NFT,
                                 'ok', markerfacecolor='w', markeredgewidth=4)
                    plt.plot(F0seq/np.double(fs)*NFT, ls='--', lw=1)
                    
                Fformant = {}
                for p in range(1,P):
                    Fformant[p-1] = poleFrq[np.int32((p-1)*nbElPerF+mu[p])]
                    if displayStatic: plt.plot(Fformant[p-1]/np.double(fs)*NFT)
                    
                if displayStatic: plt.axis('tight') ;plt.draw()
                
                ##SXhat = np.prod(Shat, axis=0)
                ##powSXhat = np.sum(SXhat, axis=0)
                ##plt.figure()
                ##imageM(db(SXhat))
                ##dirSaveArx = str('').join([prefixBDD,
                ##                           '/result_Ot-', str(Ot),
                ##                           '_nbFP-', str(numberOfFormants),
                ##                           '_winSize-',str(windowSizeInSamples),
                ##                           '_hopsize-',str(hopsize),
                ##                            '/'])
                
                fileSaveArx = str('').join([dirSaveArx,
                                            '/', bigdatnames[filenb],
                                            '.npz'])
                
                statesInMel = np.zeros([numberOfFormants, H.size])
                for p in range(1, numberOfFormants+1):
                    nbElPerF = G[p].shape[0]-1 
                    statesInMel[p-1] = poleFrqMel[\
                        np.int32((p-1)*nbElPerF+mu[p])]
                    
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
