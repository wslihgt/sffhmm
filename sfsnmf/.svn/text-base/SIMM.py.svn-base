#!/usr/bin/python
#

# Script implementing the multiplicative rules from the following
# article:
# 
# J.-L. Durrieu, G. Richard, B. David and C. Fevotte
# Source/Filter Model for Unsupervised Main Melody
# Extraction From Polyphonic Audio Signals
# IEEE Transactions on Audio, Speech and Language Processing
# Vol. 18, No. 3, March 2010
#
# with more details and new features explained in my PhD thesis:
#
# J.-L. Durrieu,
# Automatic Extraction of the Main Melody from Polyphonic Music Signals,
# EDITE
# Institut TELECOM, TELECOM ParisTech, CNRS LTCI

# copyright (C) 2010 Jean-Louis Durrieu
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import time

from numpy.random import randn
from string import join

import scipy.linalg as la

def medianFilter(energy, scope=10):
    N = energy.size
    energyFiltered = np.copy(energy)
    for n in range(N):
        energyFiltered[n] = np.median(energy[np.maximum(n-scope,0):n+scope])
        if np.isnan(energyFiltered[n]):
            energyFiltered[n] = energy[n] 
    return energyFiltered

def db(positiveValue):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(np.abs(positiveValue))

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimensio.n
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))

def computeTMatrix(binNumber, nfft, arOrder):
    """
    computeTMatrix, computes the matrix T, which is used to
    compute the power spectrum from the AR coefficients.
    """
    return la.toeplitz(np.cos(2 * np.pi * binNumber / np.double(nfft) * np.arange(arOrder + 1)))

def computeTTensor(nbins, nfft, arOrder):
    """
    """
    T = np.zeros([arOrder + 1, arOrder + 1, nbins])
    for nbin in np.arange(nbins):
        T[:,:,nbin] = computeTMatrix(nbin, nfft, arOrder)
        
    return T

def computeSPHI(ARC, TMatrices, SIGMA):
    """
    Computes the matrix of the filter, given all the AR variables
    """
    N = ARC.shape[1]
    F = TMatrices.shape[2]
    
    SPHI = np.ones([F,N])
    
    for f in np.arange(F):
        tmpMat = np.dot(TMatrices[:,:,f], ARC)
        ## SPHI[f,:] = SIGMA * np.diag(np.dot(ARC.T, tmpMat))
        for n in np.arange(N):
            SPHI[f,n] = (SIGMA[n] ** 2) / np.maximum(np.dot(ARC[:,n].T, tmpMat[:,n]),10 ** -6)
            
    return SPHI

def computeSPHIBrute(ARC, nfft, nbins, SIGMA):
    """
    SPHI, A = computeSPHIBrute(ARC, nfft, nbins, SIGMA)
    
    Computes the matrix of the filter, given all the AR variables.
    Brute force: uses the Fourier Matrix to compute.
    
    """
    arOrder, N = ARC.shape
    arOrder = arOrder - 1
    F = nbins
    
    E = np.exp(- 1j * 2 * np.pi * np.outer(np.arange(nbins), np.arange(arOrder + 1)) / np.double(nfft))
    
    return np.outer(np.ones(F),SIGMA ** 2) / np.abs(np.dot(E,ARC)) ** 2, np.abs(np.dot(E,ARC)) ** 2

def ARIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         TMatrices,
         # number of accompaniment spectra:
         numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         SIGMA0=None, ARC0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True, nfft=2048):
    """
    """
    eps = 10 ** (-20)
    lambdaARC = 1.0 ## 1.0 * 10 ** -2

    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()

    # parsing the arguments
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    Fwf0, NF0 = WF0.shape
    P1TMatrices, P2TMatrices, FTMatrices = TMatrices.shape

    if P1TMatrices != P2TMatrices:
        raise ValueError("Two first dimensions of TMatrices should be the same.")

    arOrder = P1TMatrices - 1
    
    if F != Fwf0 or F != FTMatrices:
        raise ValueError("One of WF0 or TMatrices does not have required number of frequency bins.")
    
    if SIGMA0 is None:
        SIGMA0 = np.abs(randn(N))
    else:
        if np.array(SIGMA0).shape[0] == N:
            SIGMA0 = np.array(SIGMA0)
        else:
            print "Wrong dimensions for given SIGMA0, \n"
            print "random initialization used instead"
            SIGMA0 = np.abs(randn(N))
    SIGMA = SIGMA0
    SIGMA = np.ones(SIGMA.shape) * 0.0001 # to avoid energy problems
    
    if ARC0 is None:
        ARC0 = np.abs(randn(arOrder + 1, N))
    else:
        if np.array(ARC0).shape[0] == arOrder + 1 and np.array(ARC0).shape[1] == N:
            ARC0 = np.array(ARC0)
        else:
            print "Wrong dimensions for given ARC0, \n"
            print "random initialization used instead"
            ARC0 = randn(arOrder + 1, N)
    ARC = ARC0
    ## print arOrder
    
    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(NF0, N))
    HF0 = HF00

    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
    HM = HM0

    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
    WM = WM0
    
    # Iterations to estimate the SIMM parameters:
    SF0 = np.maximum(np.dot(WF0, HF0),eps)
    SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
    SM = np.dot(WM, HM)
    hatSX = np.maximum(SF0 * SPHI + SM,eps)

    ## SX = SX + np.abs(randn(F, N)) ** 2
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])

    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)
        h2 = plt.figure(2)
        h3 = plt.figure(3)
        
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        if displayEvolution:
            h1.clf();plt.figure(1);imageM(db(HF0),cmap='jet');plt.colorbar()
            plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))]);plt.draw();

        # updating HF0:
        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)

        # This to enable octave control
        HF0[np.arange(12 * stepNotes, NF0), :] \
           = HF0[np.arange(12 * stepNotes, NF0), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes,
                                        NF0)].T, tempNumFbyN) \
                / np.maximum(
            np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
                   tempDenFbyN) \
            + lambdaHF0 * (- (alphaHF0 - 1.0) \
                           / np.maximum(HF0[
            np.arange(12 * stepNotes, NF0), :], eps) \
                           + HF0[
            np.arange(NF0 - 12 * stepNotes), :]),
            eps)) ** omega

        HF0[np.arange(12 * stepNotes), :] \
           = HF0[np.arange(12 * stepNotes), :] \
             * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                      tempNumFbyN) /
               np.maximum(
                np.dot(WF0[:, np.arange(12 * stepNotes)].T,
                       tempDenFbyN), eps)) ** omega

##        # normal update rules:
##        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) /
##                     np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HF0   : ",
            print recoError[counterError] - recoError[counterError - 1]
        counterError += 1

        # updating SIGMA
##        tempNumFbyN = (SF0 * SX) / np.maximum(A * hatSX ** 2, eps)
##        tempDenFbyN = SF0 / np.maximum(A * hatSX, eps)

##        SIGMA = SIGMA * np.sum(tempNumFbyN, axis=0) / np.maximum(np.sum(tempDenFbyN, axis=0), eps)
        
##        SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
##        hatSX = SF0 * SPHI + SM
        
##        recoError[counterError] = ISDistortion(SX, hatSX)

##        if verbose:
##            print "Reconstruction error difference after SIGMA : ", recoError[counterError] - recoError[counterError - 1]
##        counterError += 1
    
        # updating ARC
        tempNumFbyN = (SF0 * SPHI * SX) / np.maximum(A * (hatSX ** 2), eps)
        tempDenFbyN = SF0 * SPHI / np.maximum(A * hatSX, eps)
        ## compute RprimeMat
        RprimeMat = np.dot(TMatrices[0,:,:], tempNumFbyN)
        ## compute RMat
        RMat = np.dot(TMatrices[0,:,:], tempDenFbyN)

        fnum = 0
        ARCOld = np.copy(ARC)
        ARC[:, fnum] = np.dot(np.dot(la.pinv(la.toeplitz(RprimeMat[:, fnum]) +
                                            (1 / (np.sum(ARCOld[:, fnum + 1]**2))) *
                                             lambdaARC * np.outer(ARCOld[:, fnum + 1],
                                                                  ARCOld[:, fnum + 1]) * 1.0),
                                     (la.toeplitz(RMat[:, fnum]) +
                                      0.0 * lambdaARC * np.outer(ARCOld[:, fnum]+
                                                                 ARCOld[:, fnum + 1],
                                                                 ARCOld[:, fnum]+
                                                                 ARCOld[:, fnum + 1]) *
                                      (1 / (np.sum(ARCOld[:, fnum]**2 +
                                                   ARCOld[:, fnum + 1]**2))))),
                              ARCOld[:, fnum])
        for fnum in np.arange(1, N - 1):
            ARC[:, fnum] = np.dot(np.dot(la.pinv(la.toeplitz(RprimeMat[:, fnum]) +
                                                 lambdaARC * (np.outer(ARCOld[:, fnum - 1],
                                                                       ARCOld[:, fnum - 1]) /
                                                              np.sum(ARCOld[:, fnum - 1]**2) +
                                                              np.outer(ARCOld[:, fnum + 1],
                                                                       ARCOld[:, fnum + 1]) /
                                                              np.sum(ARCOld[:, fnum + 1]**2)) * 1.0),
                                         (la.toeplitz(RMat[:, fnum]) +
                                          0.0 * lambdaARC * np.outer(ARCOld[:, fnum - 1] +
                                                                     ARCOld[:, fnum] +
                                                                     ARCOld[:, fnum + 1],
                                                                     ARCOld[:, fnum - 1] +
                                                                     ARCOld[:, fnum] +
                                                                     ARCOld[:, fnum + 1]) *
                                          (1 / (np.sum(ARCOld[:, fnum - 1]**2 +
                                                       ARCOld[:, fnum]**2 +
                                                       ARCOld[:, fnum + 1]**2))))),
                                  ARCOld[:, fnum])

        fnum = N - 1
        ARC[:, fnum] = np.dot(np.dot(la.pinv(la.toeplitz(RprimeMat[:, fnum]) +
                                             (1 / (np.sum(ARCOld[:, fnum - 1]**2)))  *
                                             lambdaARC * np.outer(ARCOld[:, fnum - 1],
                                                                  ARCOld[:, fnum - 1]) * 1.0),
                                     (la.toeplitz(RMat[:, fnum]) +
                                      0.0 * (1 / (np.sum(ARCOld[:, fnum - 1]**2 +
                                                         ARCOld[:, fnum]**2))) *
                                      lambdaARC * np.outer(ARCOld[:, fnum - 1] +
                                                                ARCOld[:, fnum] ,
                                                                ARCOld[:, fnum - 1] +
                                                                ARCOld[:, fnum]))),
                              ARCOld[:, fnum])
        
        
        SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after ARC1  : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1

        if displayEvolution:
            h2.clf();plt.figure(2);imageM(db(SPHI), cmap='jet');plt.colorbar();plt.draw();
            h3.clf();plt.figure(3);imageM(ARC, cmap='jet');plt.colorbar();plt.draw();
            ## plt.clim([np.amax(db(SPHI))-100, np.amax(db(SPHI))]);plt.draw();

        normARC = np.copy(ARC[0,:])

        ARC = np.outer(np.ones(arOrder + 1), 1 / (normARC)) * ARC

        HF0 = np.outer(np.ones(NF0), 1 / (normARC ** 2)) * HF0
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after ARC2  : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        

        if displayEvolution:
            h2.clf();plt.figure(2);imageM(db(SPHI), cmap='jet');plt.colorbar();plt.draw();
            h3.clf();plt.figure(3);imageM(ARC, cmap='jet');plt.colorbar();plt.draw();
            ## plt.clim([np.amax(db(SPHI))-100, np.amax(db(SPHI))]);plt.draw();

        ## putting the roots outside the unit circle:
        for fnum in np.arange(N):
            rootsARCfnum = np.roots(ARC[:, fnum])
            index_rootsOutUnitCircle = np.abs(rootsARCfnum) > 1
            correctiveScale = np.prod(np.abs(rootsARCfnum[index_rootsOutUnitCircle]))
            HF0[:,fnum] *= 1/correctiveScale**2
            rootsARCfnum[index_rootsOutUnitCircle] = np.conjugate(1 / rootsARCfnum[index_rootsOutUnitCircle]) ## np.exp(1j * np.angle(rootsARCfnum[np.abs(rootsARCfnum) > 1])) / (np.abs(rootsARCfnum[np.abs(rootsARCfnum) > 1]))
            ## rootsARCfnum = 1.0 * rootsARCfnum
            ARC[:, fnum] = np.poly(rootsARCfnum)
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after ARC4  : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        if displayEvolution:
            h2.clf();plt.figure(2);imageM(db(SPHI), cmap='jet');plt.colorbar();plt.draw();
            h3.clf();plt.figure(3);imageM(ARC, cmap='jet');plt.colorbar();plt.draw();
            ## plt.clim([np.amax(db(SPHI))-100, np.amax(db(SPHI))]);plt.draw();

        ## averaging (TODO: make this part a better integrated estimation scheme)
        if n>20 and False:
            ARCOld = np.copy(ARC)
            ARC[:, 0] = lambdaARC * ARCOld[:, 0] + (1 - lambdaARC) / 2.0 * (ARCOld[:, 1] + ARCOld[:, 0])
            for fnum in np.arange(1,N - 1):
                ARC[:, fnum] = lambdaARC * ARCOld[:, fnum] + (1 - lambdaARC) / 3.0 * (ARCOld[:, fnum - 1] + ARCOld[:, fnum] + ARCOld[:, fnum + 1])
            fnum = N - 1
            ARC[:, fnum] = lambdaARC * ARCOld[:, fnum] + (1 - lambdaARC) / 2.0 * (ARCOld[:, fnum - 1] + ARCOld[:, fnum])

            SPHI, A = computeSPHIBrute(ARC, nfft=nfft, nbins=F, SIGMA=SIGMA)
            hatSX = np.maximum(SF0 * SPHI + SM, eps)

            recoError[counterError] = ISDistortion(SX, hatSX)

            if verbose:
                print "Reconstruction error difference after ARC3  : ", recoError[counterError] - recoError[counterError - 1]
            counterError += 1

            if displayEvolution:
                h2.clf();plt.figure(2);imageM(db(SPHI), cmap='jet');plt.colorbar();plt.draw();
                h3.clf();plt.figure(3);imageM(ARC, cmap='jet');plt.colorbar();plt.draw();
            ## plt.clim([np.amax(db(SPHI))-100, np.amax(db(SPHI))]);plt.draw();

        # updating HM
        if n > 5 and R:
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) / np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** omega, eps)

            SM = np.dot(WM, HM)
            hatSX = np.maximum(SF0 * SPHI + SM, eps)
            
            recoError[counterError] = ISDistortion(SX, hatSX)
            
            if verbose:
                print "Reconstruction error difference after HM    : ", recoError[counterError] - recoError[counterError - 1]
            counterError += 1
        
        # updating WM, after a certain number of iterations (here, after 1 iteration)
        if n > 5 and R: # this test can be used such that WM is updated only
                  # after a certain number of iterations
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) /
                                  np.maximum(np.dot(tempDenFbyN, HM.T),
                                             eps)) ** omega, eps)
            
            sumWM = np.sum(WM, axis=0)
            WM[:, sumWM>0] = (WM[:, sumWM>0] /
                              np.outer(np.ones(F),sumWM[sumWM>0]))
            HM = HM * np.outer(sumWM, np.ones(N))
            
            SM = np.dot(WM, HM)
            hatSX = np.maximum(SF0 * SPHI + SM, eps)
            
            recoError[counterError] = ISDistortion(SX, hatSX)

            if verbose:
                print "Reconstruction error difference after WM    : ",
                print recoError[counterError] - recoError[counterError - 1]
            counterError += 1

    return ARC, SIGMA, SPHI, HF0, HM, WM, recoError

   
def SIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WGAMMA,
         # number of desired filters, accompaniment spectra:
         numberOfFilters=4, numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HGAMMA0=None, HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True):
    """
    HGAMMA, HPHI, HF0, HM, WM, recoError =
        SIMM(SX, WF0, WGAMMA, numberOfFilters=4,
             numberOfAccompanimentSpectralShapes=10, HGAMMA0=None, HPHI0=None,
             HF00=None, WM0=None, HM0=None, numberOfIterations=1000,
             updateRulePower=1.0, stepNotes=4, 
             lambdaHF0=0.00, alphaHF0=0.99, displayEvolution=False,
             verbose=True)

    Implementation of the Smooth-filters Instantaneous Mixture Model
    (SIMM). This model can be used to estimate the main melody of a
    song, and separate the lead voice from the accompaniment, provided
    that the basis WF0 is constituted of elements associated to
    particular pitches.

    Inputs:
        SX
            the F x N power spectrogram to be approximated.
            F is the number of frequency bins, while N is the number of
            analysis frames
        WF0
            the F x NF0 basis matrix containing the NF0 source elements
        WGAMMA
            the F x P basis matrix of P smooth elementary filters
        numberOfFilters
            the number of filters K to be considered
        numberOfAccompanimentSpectralShapes
            the number of spectral shapes R for the accompaniment
        HGAMMA0
            the P x K decomposition matrix of WPHI on WGAMMA
        HPHI0
            the K x N amplitude matrix of the filter part of the lead
            instrument
        HF00
            the NF0 x N amplitude matrix for the source part of the lead
            instrument
        WM0
            the F x R the matrix for spectral shapes of the
            accompaniment
        HM0
            the R x N amplitude matrix associated with each of the R
            accompaniment spectral shapes
        numberOfIterations
            the number of iterations for the estimatino algorithm
        updateRulePower
            the power to which the multiplicative gradient is elevated to
        stepNotes
            the number of elements in WF0 per semitone. stepNotes=4 means
            that there are 48 elements per octave in WF0.
        lambdaHF0
            Lagrangian multiplier for the octave control
        alphaHF0
            parameter that controls how much influence a lower octave
            can have on the upper octave's amplitude.

    Outputs:
        HGAMMA
            the estimated P x K decomposition matrix of WPHI on WGAMMA
        HPHI
            the estimated K x N amplitude matrix of the filter part 
        HF0
            the estimated NF0 x N amplitude matrix for the source part
        HM
            the estimated R x N amplitude matrix for the accompaniment
        WM
            the estimate F x R spectral shapes for the accompaniment
        recoError
            the successive values of the Itakura Saito divergence
            between the power spectrogram and the spectrogram
            computed thanks to the updated estimations of the matrices.

    Please also refer to the following article for more details about
    the algorithm within this function, as well as the meaning of the
    different matrices that are involved:
        J.-L. Durrieu, G. Richard, B. David and C. Fevotte
        Source/Filter Model for Unsupervised Main Melody
        Extraction From Polyphonic Audio Signals
        IEEE Transactions on Audio, Speech and Language Processing
        Vol. 18, No. 3, March 2010
    """
    eps = 10 ** (-6)

    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()

    # renamed for convenience:
    K = numberOfFilters
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    Fwf0, NF0 = WF0.shape
    Fwgamma, P = WGAMMA.shape

    # speeding up computations:
    WF0T = WF0.T.copy() # this gets WF0 transposed to be C contiguous as well.
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        return False # A REVOIR!!!
    if HGAMMA0 is None:
        HGAMMA0 = np.abs(randn(P, K))
    else:
        if not(isinstance(HGAMMA0,np.ndarray)): # default behaviour
            HGAMMA0 = np.array(HGAMMA0)
        Phgamma0, Khgamma0 = HGAMMA0.shape
        if Phgamma0 != P or Khgamma0 != K:
            print "Wrong dimensions for given HGAMMA0, \n"
            print "random initialization used instead"
            HGAMMA0 = np.abs(randn(P, K))

    HGAMMA = HGAMMA0
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(K, N))
    else:
        Khphi0, Nhphi0 = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print "Wrong dimensions for given HPHI0, \n"
            print "random initialization used instead"
            HPHI = np.abs(randn(K, N))
        else:
            HPHI = np.array(HPHI0)

    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(NF0, N))
    HF0 = HF00
    nbIterAdaptFormant = numberOfIterations / 2.0
    varFormant0 = NF0 / 1.0# in the beginning, allow wide spread in formant
    varFormant8 = stepNotes / 1.0# in the end, variance per note should be size of one semitone
    sigmaFormant = varFormant0 ** 2 + \
                   (varFormant8 ** 2 - varFormant0 ** 2) * \
                   np.arange(1, nbIterAdaptFormant + 1) / np.double(nbIterAdaptFormant)
    HF0fo = np.copy(HF0)
    sumHF0fo = HF0fo.sum(axis=0)
    HF0fo = HF0fo / np.outer(np.ones(NF0), sumHF0fo)
    mu = np.argmax(HF0fo, axis=0)## np.dot(np.arange(NF0), HF0fo) # weighted mean, computed as value * nb_occurrences
    sigmaFormant0 = varFormant0 ** 2
    HF0fo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(NF0), sigmaFormant0)) * \
            np.exp(- 0.5 * ((np.outer(np.arange(NF0), np.ones(N)) - \
                             np.outer(np.ones(NF0), mu)) ** 2) / \
                   np.outer(np.ones(NF0), sigmaFormant0))
    HF0 = HF0fo * HF0 / \
          np.outer(np.ones(NF0), HF0fo.max(axis=0))

    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print "Wrong dimensions for given HM0, \n"
            print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
    HM = HM0

    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print "Wrong dimensions for given WM0, \n"
            print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
    WM = WM0
    
    # Iterations to estimate the SIMM parameters:
    WPHI = np.dot(WGAMMA, HGAMMA)
    SF0 = np.dot(WF0, HF0)
    SPHI = np.dot(WPHI, HPHI)
    SM = np.dot(WM, HM)
    hatSX = SF0 * SPHI + SM

    SX = SX + np.abs(randn(F, N)) ** 2
                                       # should not need this line
                                       # which ensures that data is not
                                       # 0 everywhere. 
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])

    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)
        h2 = plt.figure(2)
        h4 = plt.figure(4)

    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        if displayEvolution:
            plt.figure(1)
            h1.clf();imageM(db(HF0));
            plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))]);plt.draw();
            ## h1.clf();
            ## imageM(HF0 * np.outer(np.ones([NF0, 1]),
            ##                       1 / (HF0.max(axis=0))));
            plt.figure(2)
            h2.clf()
            imageM(db(np.dot(HGAMMA,HPHI)));plt.draw();
            plt.figure(4);
            h4.clf()
            plt.hold(True)

        # updating HF0:
        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)

        # This to enable octave control
##        HF0[np.arange(12 * stepNotes, NF0), :] \
##           = HF0[np.arange(12 * stepNotes, NF0), :] \
##             * (np.dot(WF0[:, np.arange(12 * stepNotes,
##                                        NF0)].T, tempNumFbyN) \
##                / np.maximum(
##            np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
##                   tempDenFbyN) \
##            + lambdaHF0 * (- (alphaHF0 - 1.0) \
##                           / np.maximum(HF0[
##            np.arange(12 * stepNotes, NF0), :], eps) \
##                           + HF0[
##            np.arange(NF0 - 12 * stepNotes), :]),
##            eps)) ** omega

##        HF0[np.arange(12 * stepNotes), :] \
##           = HF0[np.arange(12 * stepNotes), :] \
##             * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
##                      tempNumFbyN) /
##               np.maximum(
##                np.dot(WF0[:, np.arange(12 * stepNotes)].T,
##                       tempDenFbyN), eps)) ** omega

        # normal update rules:
##        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) /
##                     np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        HF0 = HF0 * (np.dot(WF0T, tempNumFbyN) /
                     np.maximum(np.dot(WF0T, tempDenFbyN), eps)) ** omega
        if n < nbIterAdaptFormant:
            HF0fo = np.copy(HF0)
            sumHF0fo = HF0fo.sum(axis=0)
            HF0fo = HF0fo / np.outer(np.ones(NF0), sumHF0fo)
            mu = np.argmax(HF0fo,axis=0)# np.dot(np.arange(NF0), HF0fo) # weighted mean, computed as value * nb_occurrences
            if displayEvolution:
                plt.figure(4)
                plt.plot(mu, '.')
                plt.axis('tight')
                plt.ylim([0,NF0-1])
                plt.draw()
            mu = medianFilter(mu, scope=7)
            if displayEvolution:
                plt.hold(True)
                plt.plot(mu,'.g')
            sigmaFormant0 = sigmaFormant[n]
            HF0fo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(NF0), sigmaFormant0)) * \
                    np.exp(- 0.5 * ((np.outer(np.arange(NF0), np.ones(N)) - \
                                     np.outer(np.ones(NF0), mu)) ** 2) / \
                           np.outer(np.ones(NF0), sigmaFormant0))
            HF0 = HF0fo * HF0 / \
                  np.outer(np.ones(NF0), HF0fo.max(axis=0))
            
        
        SF0 = np.dot(WF0, HF0)
        hatSX = SF0 * SPHI + SM
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HF0   : ",
            print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
    
        # updating HPHI
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / np.maximum(np.dot(WPHI.T, tempDenFbyN), eps)) ** omega
##        HPHI = HPHI * (np.dot(WPHIT, tempNumFbyN) / np.maximum(np.dot(WPHIT, tempDenFbyN), eps)) ** omega
        sumHPHI = np.sum(HPHI, axis=0)
        HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
        HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)

        SF0 = np.dot(WF0, HF0)
        SPHI = np.dot(WPHI, HPHI)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HPHI  : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        
        # updating HM
        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)
        HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) / np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** omega, eps)

        SM = np.dot(WM, HM)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HM    : ", recoError[counterError] - recoError[counterError - 1]
        counterError += 1

        # updating HGAMMA
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        HGAMMA = np.maximum(HGAMMA * (np.dot(WGAMMA.T, np.dot(tempNumFbyN, HPHI.T)) / np.maximum(np.dot(WGAMMA.T, np.dot(tempDenFbyN, HPHI.T)), eps)) ** omega, eps)

        sumHGAMMA = np.sum(HGAMMA, axis=0)
        HGAMMA[:, sumHGAMMA>0] = HGAMMA[:, sumHGAMMA>0] / np.outer(np.ones(P), sumHGAMMA[sumHGAMMA>0])
        HPHI = HPHI * np.outer(sumHGAMMA, np.ones(N))
        sumHPHI = np.sum(HPHI, axis=0)
        HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
        HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
        
        WPHI = np.dot(WGAMMA, HGAMMA)
        SF0 = np.dot(WF0, HF0)
        SPHI = np.dot(WPHI, HPHI)
        hatSX = SF0 * SPHI + SM
        
        recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print "Reconstruction error difference after HGAMMA: ",
            print recoError[counterError] - recoError[counterError - 1]
            
        counterError += 1

        # updating WM, after a certain number of iterations (here, after 1 iteration)
        if n > 0: # this test can be used such that WM is updated only
                  # after a certain number of iterations
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) /
                                  np.maximum(np.dot(tempDenFbyN, HM.T),
                                             eps)) ** omega, eps)
            
            sumWM = np.sum(WM, axis=0)
            WM[:, sumWM>0] = (WM[:, sumWM>0] /
                              np.outer(np.ones(F),sumWM[sumWM>0]))
            HM = HM * np.outer(sumWM, np.ones(N))
            
            SM = np.dot(WM, HM)
            hatSX = SF0 * SPHI + SM
            
            recoError[counterError] = ISDistortion(SX, hatSX)

            if verbose:
                print "Reconstruction error difference after WM    : ",
                print recoError[counterError] - recoError[counterError - 1]
            counterError += 1

    return HGAMMA, HPHI, HF0, HM, WM, recoError

# Formant-Tracking AutoRegressive Filters + IMM
def FoTARFIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WPHI0, poleFrq,
         numberOfFormants,
         numberOfAmpPerFormantFreq,
         # number of desired filters, accompaniment spectra:
         numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         lambdaHF0=0.00,alphaHF0=0.99,
         displayEvolution=False, verbose=True):
    
    eps = 10 ** (-50)
    
    if displayEvolution:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
        
    # renamed for convenience:
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    Fwf0, NF0 = WF0.shape
    Fwphi, K = WPHI0.shape
    
    ## speeding up computations?
    WF0T = WF0.T.copy()    # C contiguous tranposed matrix
    WPHIT = WPHI0.T.copy() # idem
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        return False # A REVOIR!!!
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(K, N))
    else:
        Khphi0, Nhphi0 = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print "Wrong dimensions for given HPHI0, \n"
            print "random initialization used instead"
            HPHI = np.abs(randn(K, N))
        else:
            HPHI = np.array(HPHI0)
            
    nbIterAdaptFormant = numberOfIterations
    nbElPerFormant = K / numberOfFormants
    varFormant0 = nbElPerFormant / 2.0# in the beginning, allow wide spread in formant
    varFormant8 = numberOfAmpPerFormantFreq / 6.0
    # in the end, variance per formant should be size of one
    sigmaFormant = varFormant0 ** 2 + \
                   (varFormant8 ** 2 - varFormant0 ** 2) * \
                   np.arange(1, nbIterAdaptFormant + 1) / np.double(nbIterAdaptFormant)
    formantRanges = {}
    for n in range(numberOfFormants):
        formantRanges[n] = n*nbElPerFormant + np.arange(nbElPerFormant)
        HPHIfo = HPHI[formantRanges[n], :]
        sumHPHIfo = HPHIfo.sum(axis=0)
        HPHIfo = HPHIfo / np.outer(np.ones(nbElPerFormant), sumHPHIfo)
        mu = np.argmax(HPHIfo, axis=0)# np.dot(np.arange(nbElPerFormant), HPHIfo)
        # weighted mean, computed as value * nb_occurrences
        sigmaFormant0 = varFormant0 ** 2
        HPHIfo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(nbElPerFormant), sigmaFormant0)) * \
                 np.exp(- 0.5 * ((np.outer(np.arange(nbElPerFormant), np.ones(N)) - \
                                  np.outer(np.ones(nbElPerFormant), mu)) ** 2) / \
                        np.outer(np.ones(nbElPerFormant), sigmaFormant0))
        HPHI[formantRanges[n], :] = HPHIfo * HPHI[formantRanges[n], :] / \
                                    np.outer(np.ones(nbElPerFormant), HPHIfo.max(axis=0))
        ## HPHIfo * np.outer(np.ones(nbElPerFormant), sumHPHIfo)
        
    # a mapping from one formant to the other, from frequency point of view:
    mapFormantToOther = np.zeros([K, numberOfFormants])
    matDist = (np.outer(poleFrq, np.ones(K)) - \
               np.outer(np.ones(K), poleFrq)) ** 2
    for n in range(numberOfFormants):
        mapFormantToOther[:,n] = np.argmin(matDist[:,formantRanges[n]], axis=1) + \
                                 formantRanges[n][0]
    
    rangeInMapping = np.int32(np.outer(np.ones(numberOfFormants), np.arange(N)))
    rangeInRemapping = np.int32(np.outer(np.arange(numberOfFormants), np.ones(N)))
    del matDist
    
    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print "Wrong dimensions for given HF00, \n"
            print "random initialization used instead"
            HF00 = np.abs(randn(NF0, N))
    HF0 = HF00
    nbIterAdaptF0 = numberOfIterations / 2.0
    varF00 = NF0 / 1.0# in the beginning, allow wide spread in formant
    varF08 = stepNotes / 2.0# in the end, variance per note should be size of one semitone
    sigmaF0 = varF00 ** 2 + \
                   (varF08 ** 2 - varF00 ** 2) * \
                   np.arange(1, nbIterAdaptF0 + 1) / np.double(nbIterAdaptF0)
    HF0fo = np.copy(HF0)
    sumHF0fo = HF0fo.sum(axis=0)
    HF0fo = HF0fo / np.outer(np.ones(NF0), sumHF0fo)
    muF0 = np.argmax(HF0fo, axis=0)## np.dot(np.arange(NF0), HF0fo)
    # weighted mean, computed as value * nb_occurrences
    sigmaF00 = varF00 ** 2
    HF0fo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(NF0), sigmaF00)) * \
            np.exp(- 0.5 * ((np.outer(np.arange(NF0), np.ones(N)) - \
                             np.outer(np.ones(NF0), muF0)) ** 2) / \
                   np.outer(np.ones(NF0), sigmaF00))
    HF0 = HF0fo * HF0 / \
          np.outer(np.ones(NF0), HF0fo.max(axis=0))
    
    if R==0:
        HM0 = 0
        WM0 = 0
    else:
        if HM0 is None:
            HM0 = np.abs(randn(R, N))
        else:
            if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
                HM0 = np.array(HM0)
            else:
                print "Wrong dimensions for given HM0, \n"
                print "random initialization used instead"
            HM0 = np.abs(randn(R, N))
            
        if WM0 is None:
            WM0 = np.abs(randn(F, R))
        else:
            if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
                WM0 = np.array(WM0)
            else:
                print "Wrong dimensions for given WM0, \n"
                print "random initialization used instead"
            WM0 = np.abs(randn(F, R))
            
    HM = HM0
    WM = WM0
    
    # Iterations to estimate the SIMM parameters:
    WPHI = WPHI0
    SF0 = np.maximum(np.dot(WF0, HF0), eps)
    SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
    SM = np.maximum(np.dot(WM, HM), eps)
    hatSX = np.maximum(SF0 * SPHI + SM, eps)
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])
    
    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 4 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print "Reconstruction error at beginning: ", recoError[0]
    counterError = 1
    if displayEvolution:
        h1 = plt.figure(1)
        h2 = plt.figure(2)
        h3 = plt.figure(3)
        h4 = plt.figure(4)
        
    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        if verbose:
            print "iteration ", n, " over ", numberOfIterations
        if displayEvolution:
            plt.figure(1)
            h1.clf()
            #imageM(db(HF0))
            #plt.clim([np.amax(db(HF0))-100, np.amax(db(HF0))]);plt.draw();
            plt.plot(db(SX))
            plt.hold(True)
            plt.plot(db(hatSX),'r')
            plt.draw()
            ## h1.clf();
            ## imageM(HF0 * np.outer(np.ones([NF0, 1]),
            ##                       1 / (HF0.max(axis=0))));
            plt.figure(2)
            h2.clf()
            imageM(db(HPHI));plt.draw();
            plt.clim([db(HPHI).max()-100, db(HPHI).max()])
            
        # updating HF0:
        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)
        # This to enable octave control
        ## HF0[np.arange(12 * stepNotes, NF0), :] \
        ##    = HF0[np.arange(12 * stepNotes, NF0), :] \
        ##      * (np.dot(WF0[:, np.arange(12 * stepNotes,
        ##                                 NF0)].T, tempNumFbyN) \
        ##         / np.maximum(
        ##     np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
        ##            tempDenFbyN) \
        ##     + lambdaHF0 * (- (alphaHF0 - 1.0) \
        ##                    / np.maximum(HF0[
        ##     np.arange(12 * stepNotes, NF0), :], eps) \
        ##                    + HF0[
        ##     np.arange(NF0 - 12 * stepNotes), :]),
        ##     eps)) ** omega
        ## 
        ## HF0[np.arange(12 * stepNotes), :] \
        ##    = HF0[np.arange(12 * stepNotes), :] \
        ##      * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
        ##               tempNumFbyN) /
        ##        np.maximum(
        ##         np.dot(WF0[:, np.arange(12 * stepNotes)].T,
        ##                tempDenFbyN), eps)) ** omega
        
        # normal update rules:
##        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) / \
##                     np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        HF0 = HF0 * (np.dot(WF0T, tempNumFbyN) / \
                     np.maximum(np.dot(WF0T, tempDenFbyN), eps)) ** omega
        
        if n < nbIterAdaptF0:
            HF0fo = np.copy(HF0)
            sumHF0fo = HF0fo.sum(axis=0)
            HF0fo = HF0fo / np.outer(np.ones(NF0), sumHF0fo)
            muF0 = np.argmax(HF0fo, axis=0)## np.dot(np.arange(NF0), HF0fo) # weighted mean, computed as value * nb_occurrences
            if displayEvolution:
                plt.figure(h4.number)
                h4.clf()
                plt.plot(muF0, '.')
                plt.axis('tight')
                plt.ylim([0,NF0-1])
                plt.draw()
            muF0 = medianFilter(muF0, scope = 10)
            if displayEvolution:
                plt.hold(True)
                plt.plot(muF0,'.g')
            sigmaF00 = sigmaF0[n]
            HF0fo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(NF0), sigmaF00)) * \
                    np.exp(- 0.5 * ((np.outer(np.arange(NF0), np.ones(N)) - \
                                     np.outer(np.ones(NF0), muF0)) ** 2) / \
                           np.outer(np.ones(NF0), sigmaF00))
            HF0fo[-1,:] = HF0fo.max(axis=0)
            HF0 = HF0fo * HF0 / \
                  np.outer(np.ones(NF0), HF0fo[-1,:])
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        recoError[counterError] = ISDistortion(SX, hatSX)
        
        if verbose:
            print "Reconstruction error difference after HF0   : ",
            print recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        # updating HPHI
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
##        HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / \
##                       np.maximum(np.dot(WPHI.T, tempDenFbyN), eps)) ** omega
        HPHI = HPHI * (np.dot(WPHIT, tempNumFbyN) / \
                       np.maximum(np.dot(WPHIT, tempDenFbyN), eps)) ** omega
        
        # post processing : Monte Carlo Approx! yeah... almost anyway
        if n < nbIterAdaptFormant:
            if displayEvolution:
                plt.figure(h3.number)
                h3.clf()
                
            muAbsolute = np.zeros([4, N])
            
            for nn in range(numberOfFormants):
                HPHIfo = HPHI[formantRanges[nn], :]
                sumHPHIfo = HPHIfo.sum(axis=0)
                HPHIfo = HPHIfo / np.outer(np.ones(nbElPerFormant), \
                                           sumHPHIfo)
                muAbsolute[nn] = np.argmax(HPHIfo, axis=0) + \
                                 formantRanges[nn][0]
                muAbsolute[nn] = medianFilter(muAbsolute[nn], scope = 10)
                
            
            muAbsolute = muAbsolute[np.argsort(poleFrq[np.int32(muAbsolute)], axis=0),
                                    rangeInMapping]
            muNew = mapFormantToOther[np.int32(muAbsolute),
                                      rangeInRemapping]
            
            for nn in range(numberOfFormants):
                HPHIfo = HPHI[formantRanges[nn], :]
                sumHPHIfo = HPHIfo.sum(axis=0)
                HPHIfo = HPHIfo / np.outer(np.ones(nbElPerFormant), sumHPHIfo)
                ## mu = np.argmax(HPHIfo, axis=0)
                ## mu = medianFilter(mu, scope = 10)
                mu = muNew[nn] - formantRanges[nn][0]
                # np.dot(np.arange(nbElPerFormant), HPHIfo)
                # weighted mean, computed as value * nb_occurrences
                sigmaFormant0 = sigmaFormant[n]
                ## print sigmaFormant0
                HPHIfo = 1 / np.sqrt(2.0*np.pi*np.outer(np.ones(nbElPerFormant),
                                                        sigmaFormant0)) * \
                         np.exp(- 0.5 * ((np.outer(np.arange(nbElPerFormant),
                                                   np.ones(N)) - \
                                          np.outer(np.ones(nbElPerFormant),
                                                   mu)) ** 2) / \
                                np.outer(np.ones(nbElPerFormant), sigmaFormant0))
                HPHI[formantRanges[nn], :] = HPHIfo * HPHI[formantRanges[nn], :] / \
                                             np.outer(np.ones(nbElPerFormant), \
                                                      HPHIfo.max(axis=0))
                ##HPHIfo * np.outer(np.ones(nbElPerFormant), sumHPHIfo)
                if displayEvolution:
                    plt.plot(poleFrq[np.int32(mu + formantRanges[nn][0])], '.')
                    plt.axis('tight')
                    plt.ylim([0, poleFrq[-1]])
                    plt.hold(True)
                    plt.draw()
                    
        
        sumHPHI = np.sum(HPHI, axis=0)
        HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
        HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
        
        SF0 = np.maximum(np.dot(WF0, HF0), eps)
        SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)
        
        recoError[counterError] = ISDistortion(SX, hatSX)
        
        if verbose:
            print "Reconstruction error difference after HPHI  : ", \
                  recoError[counterError] - recoError[counterError - 1]
        counterError += 1
        
        if R>0:
            # updating HM
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            HM = HM * (np.dot(WM.T, tempNumFbyN) / \
                       np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** omega
            
            SM = np.maximum(np.dot(WM, HM), eps)
            hatSX = np.maximum(SF0 * SPHI + SM, eps)
            
            recoError[counterError] = ISDistortion(SX, hatSX)
            
            if verbose:
                print "Reconstruction error difference after HM    : ", \
                      recoError[counterError] - recoError[counterError - 1]
            counterError += 1
            
            # updating WM, after a certain number of iterations
            # (here, after 1 iteration)
            if n > 0: # this test can be used such that WM is updated only
                      # after a certain number of iterations
                tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = 1 / np.maximum(hatSX, eps)
                WM = WM * (np.dot(tempNumFbyN, HM.T) /
                                      np.maximum(np.dot(tempDenFbyN, HM.T),
                                                 eps)) ** omega
                
                sumWM = np.sum(WM, axis=0)
                WM[:, sumWM>0] = (WM[:, sumWM>0] /
                                  np.outer(np.ones(F),sumWM[sumWM>0]))
                HM = HM * np.outer(sumWM, np.ones(N))
                
                SM = np.maximum(np.dot(WM, HM), eps)
                hatSX = np.maximum(SF0 * SPHI + SM, eps)
                
                recoError[counterError] = ISDistortion(SX, hatSX)
                
                if verbose:
                    print "Reconstruction error difference after WM    : ",
                    print recoError[counterError] - recoError[counterError - 1]
                counterError += 1
                
    return HPHI, HF0, HM, WM, recoError

def ARPIMM(SX, W, H0=None, nbIterations=100, dispEvol=True, verbose=True):
    
    eps = 10 ** -50
    
    F, N = SX.shape
    
    P = len(W) # W is a list of basis matrices
    K = np.zeros(P) # each matrix size
    for p in range(P):
        if W[p].shape[0] == F: K[p] = W[p].shape[1]
        else: raise ValueError('Size of W[%d] not same as input SX' %(p))
    
    if H0 == None or len(H0)!=P:
        H = []
        for p in range(P):
            H.append(np.random.rand(K[p], N)**2)
    else:
        H = []
        for p in range(P):
            if H0[p].shape[0] == K[p]:
                H.append(H0[p])
            else: raise ValueError("Size of H0[%d] not same as W[%d]" %(p, p))
    
    S = []
    for p in range(P):
        S.append(np.dot(W[p],H[p]))
    
    if dispEvol:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
        plt.figure(1);
        plt.clf()
        for p in range(P):
            plt.subplot(P,1,p+1)
            plt.plot(H[p])
            ##imageM(db(H[p]))
            ##if p!=0:
            ##    plt.clim([-30,0])
            ##plt.colorbar()
        plt.draw()
    
    hatSX = np.maximum(np.prod(S, axis=0), eps)
    
    recoError = np.zeros(nbIterations*P)
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose: print recoError[0]
    
    tempNum = np.zeros([F,N])
    tempDen = np.zeros([F,N])
    for n in range(nbIterations):
        if verbose: print 'iteration', n
        for p in range(P):
            tempNum = np.prod(S[:p], axis=0) * \
                      np.prod(S[(p+1):], axis=0) * \
                      SX / np.maximum((hatSX ** 2), eps)
            tempDen = np.prod(S[:p], axis=0) * \
                      np.prod(S[(p+1):], axis=0) / hatSX
            
            H[p] *= np.dot(W[p].T, tempNum) / \
                    np.maximum(np.dot(W[p].T, tempDen), eps)
            
            if p!= 0:
                sumH = H[p].sum(axis=0)
                H[p] = H[p] / np.outer(np.ones(K[p]), sumH)
                H[0] = H[0] * np.outer(np.ones(K[0]), sumH)
                S[0] = np.dot(W[0],H[0])
            
            S[p] = np.dot(W[p],H[p])
            hatSX = np.maximum(np.prod(S, axis=0), eps)
            
            recoError[n*P+p] = ISDistortion(SX, hatSX)
            if verbose: print 'error after update nb ', n*P+p, ':', recoError[n*P+p]
        
        if dispEvol:
            plt.figure(1)
            plt.clf()
            for p in range(P):
                plt.subplot(P,1,p+1)
                ##imageM(db(H[p]))
                plt.plot((H[p]))
                ##if p!=0:
                ##   plt.clim([-30, 0])
                ## plt.colorbar()
            plt.draw()
            plt.figure(2)
            plt.clf()
            imageM(db(hatSX))
            plt.draw()
                
    return H, recoError

def SparARPIMM(SX, W, poleFrq, stepNotes, H0=None, nbIterations=100,
               dispEvol=True, verbose=True):
    
    eps = 10 ** -50
    
    F, N = SX.shape
    
    P = len(W) # W is a list of basis matrices
    K = np.zeros(P) # each matrix size
    for p in range(P):
        if W[p].shape[0] == F: K[p] = W[p].shape[1]
        else: raise ValueError('Size of W[%d] not same as input SX' %(p))
    
    if H0 == None or len(H0)!=P:
        H = []
        for p in range(P):
            H.append(np.random.rand(K[p], N)**2)
    else:
        H = []
        for p in range(P):
            if H0[p].shape[0] == K[p]:
                H.append(H0[p])
            else: raise ValueError("Size of H0[%d] not same as W[%d]" %(p, p))
    
    S = []
    for p in range(P):
        S.append(np.dot(W[p],H[p]))
    
    if dispEvol:
        import matplotlib.pyplot as plt
        from imageMatlab import imageM
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
        plt.figure(1);
        plt.clf()
        for p in range(P):
            plt.subplot(P,1,p+1)
            imageM(db(H[p]))
            if p!=0:
                plt.clim([-30,0])
            plt.colorbar()
        plt.draw()
    
    hatSX = np.maximum(np.prod(S, axis=0), eps)
    
    recoError = np.zeros(nbIterations*P)
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose: print recoError[0]
    
    tempNum = np.zeros([F,N])
    tempDen = np.zeros([F,N])
    for n in range(nbIterations):
        if verbose: print 'iteration', n
        for p in range(P):
            tempNum = np.prod(S[:p], axis=0) * \
                      np.prod(S[(p+1):], axis=0) * \
                      SX / np.maximum((hatSX ** 2), eps)
            tempDen = np.prod(S[:p], axis=0) * \
                      np.prod(S[(p+1):], axis=0) / hatSX
            
            H[p] *= np.dot(W[p].T, tempNum) / \
                    np.maximum(np.dot(W[p].T, tempDen), eps)
            
            
            
            if p!= 0:
                sumH = H[p].sum(axis=0)
                H[p] = H[p] / np.outer(np.ones(K[p]), sumH)
                H[0] = H[0] * np.outer(np.ones(K[0]), sumH)
                S[0] = np.dot(W[0],H[0])
            
            S[p] = np.dot(W[p],H[p])
            hatSX = np.maximum(np.prod(S, axis=0), eps)
            
            recoError[n*P+p] = ISDistortion(SX, hatSX)
            if verbose: print 'error after update nb ', n*P+p, ':', recoError[n*P+p]
            
            
        if dispEvol:
            plt.figure(1)
            plt.clf()
            for p in range(P):
                plt.subplot(P,1,p+1)
                imageM(db(H[p]))
                if p!=0:
                    plt.clim([-30, 0])
                ## plt.colorbar()
            plt.draw()
            plt.figure(2)
            plt.clf()
            imageM(db(hatSX))
            plt.draw()
                
    return H, recoError
