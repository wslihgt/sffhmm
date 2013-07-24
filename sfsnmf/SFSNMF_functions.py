
"""
ARPIMM module: Auto-Regressive (p) Instantaneous Mixture Model
--------------------------------------------------------------

Support functions and main estimation algorithm for the ARPIMM model,
also called the Source/Filter Sparse Non-Negative Matrix Factorization
(SFSNMF), as published in the following articles:

 Durrieu, J.-L. and Thiran, J.-P.
 \"Sparse Non-Negative Decomposition Of Speech Power Spectra
 For Formant Tracking\"
 proc. of the IEEE International Conference on Acoustics, Speech and
 Signal Processing, Prague, Czech Republic, 2011.

 Durrieu, J.-L. and Thiran, J.-P.
 \"Source/Filter Factorial Hidden Markov Model,
 with Application to Pitch and Formant Tracking\"
 IEEE Transactions on Audio, Speech and Language Processing,
 Submitted Jan. 2013, Accepted July 2013.

Copyright (c) 2010 - 2013 Jean-Louis Durrieu
http://www.durrieu.ch/research/

"""

import numpy as np
import os
import warnings

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
    
    odgd /= np.abs(odgd).max() # added so that less noise after in estimation
    
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
    """Generates the AR frequency responses for the desired
    pole amplitude and frequency ranges.
    
    INPUT
    -----
     numberFrequencyBins
      the number of frequency bins to be kept, i.e. the size of the frequency
      domain vectors/frequency response vectors.
      
     sizeOfFourier
      the size of the Fourier transform
      
     Fs
      the sampling rate
      
     formantsRange
      a list of two frequency values: the lower and upper bound for a given
      formant
      
     bwRange
      a list of two bandwidth values, the lower and upper bound for a given
      formant. If None is given, then the `maxF0` argument is used as the
      minimum allowed bandwidth (otherwise, the generated spectral shapes could
      be misinterpreted as spectral peaks due to voiced sources), while the
      widest bandwidth is set to 10% of the sampling rate.
      
     numberOfAmpsPerPole
      the number of amplitudes to generate within the desired range.
      
     numberOfFreqPerPole
      the number of pole frequencies to generate from the provided range.
      
     maxF0
      the maximum F0 frequency, for the source/filter model.


    OUTPUT
    ------
     bwRange
      ndarray of all the bandwidth values
      
     freqRanges
      ndarray of all the frequency values
      
     poleAmp
      the actual values of the amplitudes for each pole
     
     poleFrq
      the frequency values of the pole
     
     WGAMMA
      F x K ndarray
      The dictionary matrix containing the spectral frequency responses.
      Each column is a frequency response for a given formant/complex pole:
      
    
    """
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

def genARbasis(numberFrequencyBins, sizeOfFourier, Fs, 
               formantsRange=None, 
               bwRange=None, 
               numberOfAmpsPerPole=5, 
               numberOfFreqPerPole=60,
               maxF0 = 1000.0):
    if formantsRange is None:
        formantsRange = {}
        formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
        formantsRange[1] = [ 550.0, 3500.0]
        formantsRange[2] = [1400.0, 4500.0]
        formantsRange[3] = [2400.0, 6000.0] # adding one for full band
        formantsRange[4] = [3300.0, 8000.0]
        formantsRange[5] = [4500.0, 8000.0]
        formantsRange[6] = [5500.0, 8000.0]
        if Fs != 16000:
            warnings.warn('The sampling rate is not 16kHz, but %d.\n' %Fs +
                          'Please check that the provided formantRange is ' +
                          'suitable for that particular sampling rate:\n' +
                          str(formantRange))
    
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
