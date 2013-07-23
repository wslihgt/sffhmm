#!/usr/bin/python
#

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
import time
import warnings

from numpy.random import randn
from string import join

import scipy.linalg as la

####################
# Support functions
####################

def medianFilter(energy, scope=10):
    """Median filter: outputs a smoothed
    version of a given sequence, by replacing each sample
    by the median value over a `2*scope+1`-long window, centered around
    the current sample.
    """
    N = energy.size
    energyFiltered = np.copy(energy)
    for n in range(N):
        energyFrame = energy[np.maximum(n-scope,0):n+scope]
        energyFiltered[n] = np.median(energyFrame[energyFrame>0])
        if np.isnan(energyFiltered[n]):
            energyFiltered[n] = energy[n] 
    return energyFiltered

def db(positiveValue):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    
    NB: assumes the input is positive, and does not assume
    whether it is an energy or an amplitude. 
    """
    return 10 * np.log10(np.abs(positiveValue))

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))

def normalDistribution(vector, mu, sigma):
    """real normal distribution function, for a given vector, and with provided
    parameters
    """
    return 1 / (np.sqrt(2.0 * np.pi * sigma**2))\
           * np.exp(- ((vector - mu) ** 2) / (2.0 * sigma ** 2))

def normalDistributionMat(matrix, muMat, sigma):
    """returns the probability, for each value of `matrix`, assuming
    it was distributed as a real normal variable, with provided parameters. 
    """
    return (1 /
            (np.sqrt(2.0 * np.pi * np.outer(np.ones(matrix.shape[0]),
                                            sigma**2)))
            * np.exp(- ((matrix - np.outer(np.ones(matrix.shape[0]),
                                           muMat)) ** 2)
                     / (2.0 * np.outer(np.ones(matrix.shape[0]),
                                       sigma**2))))

# from scikits.learn.mixture, but without checking the min value:
def normalize(A, axis=None):
    """normalize(A, axis=None)
    
    Normalizes the array A by dividing its elements such that
    the sum over the given axis equals 1, except if that sum is 0,
    in which case, it ll stay 0.
    
    Parameters
    ----------
    A : ndarray
        An array containing values, to be normalized.
    axis : integer, optional
        Axis over which the ndarray is normalized. By default, axis
        is None, and the array is normalized such that its sum becomes 1.
    
    Returns
    -------
    out : ndarray
        The normalized ndarray.
    
    See also
    --------
    scikits.learn.normalize

    Examples
    --------
    >>> normalize(np.array([[0., 1.], [0., 5.]]))
    array([[ 0.        ,  0.16666667],
           [ 0.        ,  0.83333333]])
    >>> normalize(np.array([[0., 1.], [0., 5.]]), axis=0)
    array([[ 0.        ,  0.16666667],
           [ 0.        ,  0.83333333]])
    
    """
    Asum = A.sum(axis)
    if not(axis is None) and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    else:
        if Asum==0:
            Asum  = 1
    return A / Asum

def SFSNMF(SX,
           W, WR, stepNotes,
           G0=None,
           H0=None,
           nbIterations=200,
           scopeMedian=10,
           dispEvol=True, verbose=True,
           poleFrq=None, poleAmp=None, smoothIt=True):
    """SFSNMF
    source/filter sparse non-negative matrix factorization
    
    The aim of this algorithm is to estimate a tensor G that minimizes
    the Itakura Saito distance between the matrix `SX` (short-term power
    spectrum of the input audio signal) and the matrix:

    .. math::

        H \bullet (WR GR) \bullet \prod_{p=0}^P (W[p] G[p])

    In the following, we assume that the number of frequency bins is F, the
    number of frames is N. There are P elements for the Source/Filter model,
    that is 1 source and P-1 formants. 

    INPUTS
    ------
     SX F x N ndarray
      observation short-term power spectrum
     
     W list of P spectral dictionaries
      For each p in [0, P-1]:
      W[p] is a F x K[p] ndarray of spectral dictionary for the source
      component (p=0) and for the formants (p>0). 

     WR F x KR ndarray
      spectral dictionary, consisting of KR smooth elements, used to
      define the recording condition filter. 

     stepNotes int
      

     G0 list of P ndarray
      

     H0 N ndarray
      initial vector for the energy component of the model

     nbIterations

     scopeMedian

     dispEvol

     verbose

     poleFrq

     poleAmp

     smoothIt

    OUTPUTS
    -------
     G
      List of estimated amplitude matrices 

     GR
      Amplitude matrix for the recording 

     H

     recoError
    
    REFERENCE
    ---------
    revised version from ICASSP 2011, to journal article on FHMM:
    
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
     
    
    """
    
    eps = 10 ** -50
    
    F, N = SX.shape
    
    # parsing W, list of matrices of spectral shapes
    # one for each formant
    #    W[0] should be the source spectral shapes
    P = len(W) # W is a list of basis matrices
    K = np.zeros(P) # each matrix size
    for p in range(P):
        if W[p].shape[0] == F: K[p] = W[p].shape[1]
        else: raise ValueError('Size of W[%d] not same as input SX' %(p))
    
    if WR.shape[0] == F: KR = WR.shape[1]
    else: raise ValueError('Size of WR not same as input SX')
    
    # parsing initial G0
    if G0 == None or len(G0)!=P:
        G = {}
        warnings.warn("Provided G0 None or badly initialized.")
        for p in range(P):
            G[p] = np.random.rand(K[p], N)**2
            G[p] = normalize(G[p], axis=0)
    else:
        G = {}
        for p in range(P):
            if G0[p].shape[0] == K[p]:
                G[p] = normalize(G0[p], axis=0)
            else: raise ValueError("Size of G0[%d] not same as W[%d]" %(p, p))
    
    # energy parameters:
    if H0 is None or H0.size != N:
        H = SX.sum(axis=0)
    else:
        H = np.copy(H0)
    # recording condition parameters:
    GR = np.ones(KR)
    
    # a list of intermediate component matrices
    S = []
    for p in range(P):
        S.append(np.dot(W[p],G[p]))
    
    # displaying 
    if dispEvol:
        import matplotlib.pyplot as plt
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.origin'] = 'lower'
        plt.ion()
        print "Is the display interactive? ", plt.isinteractive()
        plt.figure(1);
        plt.clf()
        for p in range(P):
            plt.subplot(P,1,p+1)
            plt.imshow(db(G[p]))
            if p!=0:
                plt.clim([-30,0])
            plt.colorbar()
        plt.draw()
        
    
    nbUncstIter = 0
    nbEndIter = 5
    nbIterSig = np.maximum(nbUncstIter+nbEndIter, \
                           nbIterations - (nbUncstIter+nbEndIter)) 
    # a sequence of sigmas for the gauss model for G
    sigma = np.zeros(nbIterSig)
    sigmaInf = 9.0
    sigma0 = (K.max()) ** 2
    # log of sigma linearly decreasing over iterations
    for n in range(nbIterSig):
        sigma[n] = np.exp(np.log(sigma0) + \
                          (np.log(sigmaInf) - \
                           np.log(sigma0)) / \
                          (nbIterSig - 1.0) * n)
    
    # hatSX is the estimated SX
    #    energy * recordFilter * source * vocalFilter
    hatSX = H * np.vstack(np.dot(WR,GR)) * np.prod(S, axis=0)
    hatSX = np.maximum(hatSX, eps)
    
    if dispEvol:
        # display also initial estimate
        plt.figure(2)
        plt.clf()
        plt.imshow(db(hatSX))
        plt.draw()
    
    recoError = np.zeros(nbIterations*P)
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose: print "Initial error: "+str(recoError[0])
    
    # useful temporary numerator and denominator
    # for the NMF algorithm
    tempNum = np.zeros([F,N])
    tempDen = np.zeros([F,N])
    
    for n in range(nbIterations):
        if verbose: print 'iteration', n, ' of ', nbIterations
        for p in range(P):
            tempNum = SX / np.maximum((hatSX * S[p]), eps)
            tempDen = 1 / np.maximum(S[p], eps)
            
            G[p] = G[p] * np.dot(W[p].T, tempNum) / \
                    np.maximum(np.dot(W[p].T, tempDen), eps)
            
            # take muG as the barycenter (mean of the distribution):
            if n >= nbUncstIter and n < nbIterations-nbEndIter:
                ncstIter = n-nbUncstIter
                if p==0:
                    muG = np.dot(np.arange(K[p]-1) * \
                                 (np.arange(K[p]-1, 0, -1))**2, \
                                 G[p][:-1,:]) / \
                                 np.dot((np.arange(K[p]-1, 0, -1))**2,
                                        np.maximum(G[p][:-1,:], eps))
                else:
                    muG = np.dot(np.arange(K[p]-1), G[p][:-1,:]) / \
                          np.sum(np.maximum(G[p][:-1,:], eps), axis=0)
                if dispEvol:
                    plt.figure(1)
                    plt.subplot(P+2,1,p+1)
                    plt.plot(muG, 'g')
                    plt.axis('tight')
                    plt.draw()
                
                # smooth the obtained sequence:
                if smoothIt:
                    muG = medianFilter(muG, scope=scopeMedian)
                
                if dispEvol:
                    plt.figure(1)
                    plt.subplot(P+2,1,p+1)
                    plt.plot(muG, 'k')
                    plt.axis('tight')
                    plt.draw()
                
                # weights to get sparse:
                Gp = np.exp(- 0.5 * ((np.outer(np.arange(K[p]),
                                               np.ones(N)) - \
                                      np.outer(np.ones(K[p]),
                                               muG)) ** 2) / \
                            np.outer(np.ones(K[p]), sigma[ncstIter]))
                if True or p==0:
                    # take into account last el. in W[p],
                    # which corresponds to the
                    # unvoiced element or flat freq. response. 
                    Gp[-1,:] = Gp.max(axis=0)
            
                G[p] = Gp * G[p] / \
                       np.outer(np.ones(K[p]), Gp.max(axis=0))
            
            sumG = G[p].max(axis=0)# G[p].sum(axis=0)
            G[p] = G[p] / np.outer(np.ones(K[p]), sumG)
            H = H * sumG
            
            S[p] = np.dot(W[p],G[p])
            hatSX = H * np.vstack(np.dot(WR,GR)) * np.prod(S, axis=0)
            hatSX = np.maximum(hatSX, eps)
            
            # updating recording condition filter
            tempNum = SX / np.maximum(hatSX * np.vstack(np.dot(WR, GR)), eps)
            tempDen = N / np.maximum(np.dot(WR, GR), eps)
            GR = GR * np.dot(WR.T, np.sum(tempNum, axis=1)) / \
                 np.maximum(np.dot(WR.T, tempDen), eps)

            sumG = GR.sum()
            GR = GR / sumG
            H = H * sumG
            
            hatSX = H * np.vstack(np.dot(WR,GR)) * np.prod(S, axis=0)
            hatSX = np.maximum(hatSX, eps)
            
            # updating energy component
            H = np.mean(SX / np.maximum(np.vstack(np.dot(WR,GR)) * \
                                        np.prod(S, axis=0), eps), \
                        axis=0)
            
            hatSX = H * np.vstack(np.dot(WR,GR)) * np.prod(S, axis=0)
            hatSX = np.maximum(hatSX, eps)
            
            recoError[n*P+p] = ISDistortion(SX, hatSX)
            if verbose:
                print 'error after update nb ', n*P+p, \
                      ':', recoError[n*P+p], ' evol compared with previous: ', \
                      recoError[n*P+p] - recoError[n*P+p-1]
            
        if dispEvol:
            plt.figure(1)
            plt.clf()
            for p in range(P):
                plt.subplot(P+2,1,p+1)
                plt.imshow(db(np.maximum(G[p],eps)))
                if p!=0:
                    plt.clim([-30, 0])
                ## plt.colorbar()
            plt.subplot(P+2,1,P+1)
            plt.plot(H)
            plt.subplot(P+2,1,P+2)
            plt.plot(db(np.dot(WR,GR)))
            plt.draw()
            plt.figure(2)
            plt.clf()
            plt.imshow(db(hatSX))
            plt.draw()
        
    return G, GR, H, recoError
    
