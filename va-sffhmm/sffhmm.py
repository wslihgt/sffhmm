# copyright (C) 2010 - 2013 Jean-Louis Durrieu
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
"""
Factorial Hidden Markov Model and Source/Filter FHMM classes
------------------------------------------------------------

Support code for the article

 Jean-Louis Durrieu and Jean-Philippe Thiran,
 \"Source/Filter Factorial Hidden Markov Model, 
 with Application to Pitch and Formant Tracking\", 
 submitted Jan. 2013, accepted July 2013.

After
.. [Gha97] Z. Ghahramani and M. Jordan,
           Factorial Hidden Markov Models
           Machine Learning 29, 245-273 (1997)

based on the Matlab code at:
http://mlg.eng.cam.ac.uk/zoubin/software/fhmm.tar.gz
and with a few lines coming from scikits.learn.hmm (v 0.11)

Note
----

 The forward/backward procedure in C is used in scipy.weave.inline.
 This is mostly a bad idea. It works so far, but weave is discontinued.
 Should switch to cython or swig.

"""
# Factorial Hidden Markov Models
#
# Author: Jean-Louis Durrieu <jean-louis@durrieu.ch>
# 2012


import string
import os 

import numpy as np

from numpy import prod as npprod
from numpy import sum as npsum
from numpy import int32 as npint
from numpy import arange as nparange

from scipy import linalg
# non-negative least square solver:
from scipy.optimize import nnls
from scipy.signal import lfilter

import warnings

# from scikits.learn (sklearn) v0.9:
import sklearn
from sklearn.hmm import _BaseHMM, GaussianHMM
from sklearn.mixture import (_distribute_covar_matrix_to_match_cvtype,
                             _validate_covars)
from sklearn.base import BaseEstimator
# TODO: check that this version checker is actually working the right way:
if sklearn.__version__.split('.')[1] in ('9','10','8'):
    from sklearn.utils.extmath import logsum
else:
    warnings.warn("This script builds on top of scikits.learn.hmm"+
                  " and was developed under version 0.8.\n"+
                  "It should work with version 0.8, 0.9 and 0.10.\n"+
                  "Because of changes in scikits.learn.hmm below and after,\n"+
                  "the results can't be guaranteed.\n"+
                  "Your version is sklearn v%s." %str(sklearn.__version__))
    from sklearn.utils.extmath import logsumexp as logsum

# home brew forward backward, with inline:
"TODO: use proper import of C functions, like Cython or SWIG"
from scipy.weave import inline
fwdbwd_filename = 'computeLogDensity_FB_Viterbi.c'
fwdbwd_file = open(fwdbwd_filename,'r')
fwdbwd_supportcode = fwdbwd_file.read()
fwdbwd_file.close()

try:
    import LPCanalysis as lpcana
    withLPC = True
except ImportError:
    warning.warn("Could not load LPCanalysis, not allowing lpc init for sffhmm")
    withLPC = False

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

# from scikits.learn.mixture, but without checking the min value:
def normalize(A, axis=None):
    """normaliwe(A, axis=None)
    
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

class _BaseFHMM(BaseEstimator): # TODO: should it be subclass of _BaseHMM?
    """ Base class for Factorial HMM
    """
    def __init__(self, n_states=[2,2], 
                 startprob=None, transmat=None,
                 startprob_prior=None,
                 transmat_prior=None,
                 HMM=None):
        """Base class for the Factorial HMM isntances
        
        It is an sklearn BaseEstimator subclass. However, 

        INPUTS
        ------
         n_states : list
          a list of integers: n_states[p] is the number of states for the p^th
          independent HMM chain.
          
         startprob : None or list
          list of ndarray (vector) starting probabilities for each of the
          states and for each HMM chain
          
         transmat : None or list
          list of ndarray (matrix) for the transition probabilities, for each
          HMM chain, between the different states of the given chain. 
         
         startprob_prior : None or list
          list of startprob priors, see the 

         transmat_prior : None or list
          
         
         HMM :
         
         
        Remarks
        -------
         2013-07-23:
         Implements the decode_var method. In line with the sklearn module,
         the `decode` method aims at decoding the sequence, that is to
         provide the sequence of most likely states given a (HMM) model and
         the observation matrix.
         The provided `decode_var` method actually corresponds to the
         variational approximation of the decoding.

         It is an sklearn BaseEstimator subclass. However, it does not
         completely comply with the sklearn directives yet. We had to use
         a `decode_var` method instead of the `decode` method, in order
         to pass it arguments. It would seem that this distinction is not
         necessary anymore, and one should work to make this compliant with
         more recent sklearn directives.
         
         TODO: Actually, the FHMM is also an HMM, and allowing it to subclass
         the _BaseHMM class would be good, for instance to be able to compute
         exact inference whenever possible (few states). The transition matrix
         can be infered at this level, but for other parameters (for instance
         mean and covariance parameters for GaussianHMM) need to be addressed
         in subclasses, depending on the chosen underlying observation
         likelihood.

        """
        
        self._n_states = list(n_states)
        self._n_chains = len(self._n_states)
        # number of states, but not all possible states,
        # only sum of number of states for each chain:
        self._n_states_all = npsum(n_states)
        
        ## should also probably check the length of all the arguments
        ## before assigning to self.HMM...
        if startprob is None or \
               len(startprob)!=self._n_chains:
            startprob = [None] * self._n_chains
            
        if transmat is None or \
               len(transmat)!=self._n_chains:
            transmat = [None] * self._n_chains
        
        if startprob_prior is None or \
               len(startprob_prior)!=self._n_chains:
            startprob_prior = [None] * self._n_chains
        
        if transmat_prior is None or \
               len(transmat_prior)!=self._n_chains:
            transmat_prior = [None] * self._n_chains        
        
        self.HMM = {}
        
        for n in range(self._n_chains):
            self.HMM[n] = _BaseHMM(n_states[n], startprob[n], transmat[n],
                                   startprob_prior=startprob_prior[n],
                                   transmat_prior=transmat_prior[n])
        
        # to avoid getting an error when printing the object:
        self.startprob = None
        self.transmat = None
        self.startprob_prior = None
        self.transmat_prior = None
        
    def eval_var(self, obs, n_innerLoop=10,
                 maxrank=None, beamlogprob=-np.Inf,
                 tol=0.0001):
        """ TODO: to be tested and corrected according to decode_var"""
        obs = np.asanyarray(obs)
        nframes = obs.shape[0]
        
        ## posteriors = np.zeros(nframes, self.n_states_all)
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains):
            posteriors[n] = np.ones(self._n_states[n], nframes)/\
                            (1.*self._n_states[n])
            logPost[n] = np.zeros(self._n_states[n], nframes)
        
        for i in range(n_innerLoop):
            # for stopping condition
            posteriors0 = list(posteriors)
            logPost0 = list(logPost)
            # compute variational parameters as in [Gha97]
            # "% first compute h values based on mf"
            frameVarParams = self._compute_var_params(obs, posterior)
            # Forward Backward for each chain:
            for n in range(self.n_chains):
                ## idxStates = npsum(self.n_states[:n])+\
                ##             nparange(self.n_states[n])
                ## idxStates = npint(idxStates)
                dumpLogProba, fwdLattice = self.HMM[n]._do_forward_pass(\
                                                    frameVarParams[n],
                                                    maxrank,
                                                    beamlogprob)
                del dumpLogProba
                bwdlattice = self.HMM[n]._do_backward_pass(\
                                                    frameVarParams[n],
                                                    fwdlattice,
                                                    maxrank,
                                                    beamlogprob)
                gamma = fwdlattice + bwdlattice
                logPost[n] = gamma - logsum(gamma, axis=1)
                posteriors[n] = np.exp(logPost[n]).T
            
            delmf = npsum(npsum(np.concatenate(posteriors) * \
                                np.concatenate(logPost))) \
                  - npsum(npsum(np.concatenate(posteriors) * \
                                np.concatenate(logPost0)))
            if delmf < nframes*tol:
                itermf = i
                break
        
        return posteriors # should return also logL: approximate of it?
    
    def decode_var(self, obs, n_innerLoop=10, maxrank=None,
                   beamlogprob=-np.Inf, verbose=False, debug=False,
                   n_repeatPerChain=None, postInitMeth='random'):
        """decode_var
        
        returns most likely seq of states, and
        the posterior probabilities.
        
        For now, the sequence of states is calculated through the
        variational approximation, i.e. we first compute the variational
        parameters and the corresponding posteriors for each chain, iterate
        over that process n_innerLoop times, and at last outputs the states
        computed through Viterbi for each chain, on the log variational
        parameters as <<log likelihood>> (as suggested in [Gha97]).
        
        Important I : this is not the true sequence (should run Viterbi on all
        states, with true likelihood). It is possible to do, but very expensive
        in terms of memory (especially for source/filter subclass).
        
        """
        obs = np.asanyarray(obs)
        
        nframes = obs.shape[0]
        
        if n_repeatPerChain is None:
            n_repeatPerChain = [1] * self._n_chains
        if len(n_repeatPerChain) != self._n_chains:
            print "Warning: the provided n_repeatPerChain arg has wrong length."
            print "         Falling to default: only one repeat for each chain."
            n_repeatPerChain = [1] * self._n_chains
        
        ## First compute the variational parameters
        if verbose:
            print "Computing the variational parameters"
        fwdlattice = {}
        bwdlattice = {}
        
        posteriors, logPost = self._init_posterior_var(obs=obs,
                                                       method=postInitMeth,
                                                       debug=debug,
                                                       verbose=verbose)
        
        logH = self._compute_energyComp(obs=obs,
                                        posteriors=posteriors,
                                        debug=debug)
        
        if debug:
            plt.figure(200)
            plt.clf()
            plt.imshow(np.concatenate(logPost.values(), axis=1).T)
            plt.colorbar()
            plt.title('initial log posteriors')
            plt.draw()
        
        posteriors, logPost = self._init_posterior_var(obs=obs-np.vstack(logH),
                                                       method=postInitMeth,
                                                       debug=debug,
                                                       verbose=verbose)
        
        if debug:
            plt.figure(200)
            plt.clf()
            plt.imshow(np.concatenate(logPost.values(), axis=1).T)
            plt.colorbar()
            plt.title('initial log posteriors')
            plt.draw()
        
        for inn in range(n_innerLoop):
            if verbose:
                print "    inner loop for variational approx.", inn,\
                      "over", n_innerLoop
            # for stopping condition
            posteriors0 = dict(posteriors)
            logPost0 = dict(logPost)
            for n in range(self._n_chains):
                for t in range(n_repeatPerChain[n]):
                    # computing log variational parameters
                    if verbose:
                        print "        chain number:", n, "in", self._n_chains
                    frameLogVarParams = self._compute_var_params_chain(
                        obs=obs,
                        #chge here for alt. est. strategy
                        posteriors=posteriors, 
                        chain=n, debug=debug)
                    # computing the forward/backward passes on the variational
                    # parameters, for the given chain
                    if debug:
                        print "        Forward backward for posteriors"
                    fwdlattice[n], bwdlattice[n] = \
                                    self._do_fwdbwd_pass_var_hmm_c(\
                                                           frameLogVarParams,
                                                           chain=n)
                    # ... and deducing the posterior probas
                    gamma = fwdlattice[n] + bwdlattice[n]
                    logPost[n] = (gamma.T - logsum(gamma, axis=1)).T
                    posteriors[n] = np.exp(logPost[n])
                    if debug: print "        Done."
                    
                    if debug:
                        if verbose: print "            Display."
                        plt.figure(1)
                        plt.clf()
                        plt.imshow(frameLogVarParams-
                                   np.vstack(logsum(frameLogVarParams,axis=1)))
                        plt.colorbar()
                        plt.title("frameLogVarParams "+str(n))
                        plt.draw()
                        plt.figure(2)
                        plt.clf()
                        plt.subplot(211)
                        plt.imshow(logPost[n].T)
                        plt.colorbar()
                        plt.title("posterior "+str(n))
                        plt.subplot(212)
                        plt.imshow(bwdlattice[n].T)
                        plt.colorbar()
                        plt.title("bwdlattice "+str(n))
                        plt.draw()
                    
            # stop condition (from [Gha97] and Matlab code)
            delmf = npsum(npsum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost.values(),axis=1)))- \
                    npsum(npsum(np.concatenate(posteriors.values(),axis=1)*\
                                np.concatenate(logPost0.values(),axis=1)))
            ## print delmf#DEBUG
            if delmf < nframes*0.000001 and inn>0:
                itermf = inn
                break
        
        state_sequences = {}
        ## frameVarParams = self._compute_var_params(obs, posteriors)
        # print frameVarParams #DEBUG
        for n in range(self._n_chains):
            frameLogVarParams = self._compute_var_params_chain(\
                                                obs=obs,
                                                posteriors=posteriors,
                                                chain=n, debug=debug)
            logprob, state_sequences[n] = self.HMM[n]._do_viterbi_pass(\
                                                        frameLogVarParams,
                                                        maxrank,
                                                        beamlogprob)
            del logprob
        
        return state_sequences, posteriors
    
    def fit_var(self, obs, n_iter=100, n_innerLoop=10,
                thresh=1e-2, params=string.letters,
                init_params=string.letters,
                maxrank=None, beamlogprob=-np.Inf,
                tol=0.0001, innerTol=1e-6,
                verbose=False, debug=False,
                **kwargs):
        """fit_var
        
        n_iter is number of baum-welch cycles
        n_innerLoop number of mean field computations (struct. var. approx.)
        
        """
        self._init(obs, init_params)
        
        logprob = []
        for i in xrange(n_iter):
            if verbose:
                print "iteration", i,"over",n_iter
            # Expectation step
            stats = self._initialize_sufficient_statistics_var()
            ##print stats.keys()#DEBUG
            curr_logprob = 0
            for seq in obs:
                nframes = seq.shape[0]
                posteriors = {}
                logPost = {}
                fwdlattice = {}
                bwdlattice = {}
                for n in range(self._n_chains): 
                    # posteriors[n] = np.ones([nframes, self._n_states[n]])/\
                    #                 (1.*self._n_states[n])
                    posteriors[n] = np.random.rand(nframes,
                                                   self._n_states[n])
                    posteriors[n] = normalize(posteriors[n])
                    logPost[n] = np.log(posteriors[n])
                    # np.zeros([nframes, self._n_states[n]])
                
                for inn in range(n_innerLoop):
                    if verbose:
                        print "    inner loop for variational approx.", inn,\
                              "over",n_innerLoop
                    # for stopping condition
                    posteriors0 = dict(posteriors)
                    logPost0 = dict(logPost)
                    ## the following commented version works well for small
                    ## scale problems, but with big _n_states, one should
                    ## consider not storing all the variational params.
                    ## # compute variational parameters as in [Gha97]
                    ## frameVarParams=self._compute_var_params(seq, posteriors)
                    ## # with inline :
                    ## fwdlattice, bwdlattice = self._do_fwdbwd_pass_var_c(\
                    ##                                         frameVarParams)
                    for n in range(self._n_chains):
                        ## idxStates = npsum(self.n_states[:n])+\
                        ##             nparange(self.n_states[n])
                        ## idxStates = npint(idxStates)
                        ## dumpLogProba, fwdlattice[n] = \
                        ##     self.HMM[n]._do_forward_pass(\
                        ##                              frameVarParams[n],
                        ##                              maxrank,
                        ##                              beamlogprob)
                        ## bwdlattice[n] = self.HMM[n]._do_backward_pass(\
                        ##                                   frameVarParams[n],
                        ##                                     fwdlattice[n],
                        ##                                     maxrank,
                        ##                                     beamlogprob)
                        # 20111219T1725 change of plans: compute one by one
                        # otherwise takes too much memory
                        frameLogVarParams = self._compute_var_params_chain(\
                                                obs=seq,
                                                posteriors=posteriors,
                                                chain=n)
                        fwdlattice[n], bwdlattice[n] = \
                                                self._do_fwdbwd_pass_var_hmm_c(\
                                                                 frameLogVarParams,
                                                                 chain=n)
                        if debug:
                            plt.figure(1)
                            plt.clf()
                            plt.imshow(frameLogVarParams.T)
                            plt.title("frameLogVarParams")
                            plt.draw()
                            plt.figure(2)
                            plt.clf()
                            plt.subplot(211)
                            plt.imshow(fwdlattice[n])
                            plt.title("fwdlattice")
                            plt.subplot(212)
                            plt.imshow(bwdlattice[n])
                            plt.title("bwdlattice")
                            plt.draw()
                        
                        gamma = fwdlattice[n] + bwdlattice[n]
                        logPost[n] = (gamma.T - logsum(gamma, axis=1)).T
                        posteriors[n] = np.exp(logPost[n])
                        ## del dumpLogProba
                    ## lpr, fwdlattice = self._do_forward_pass_var(frameVarParams,
                    ##                                             maxrank,
                    ##                                             beamlogprob)
                    ## bwdlattice = self._do_backward_pass(framelogprob, fwdlattice,
                    ##                                     maxrank, beamlogprob)
                    ## gamma = fwdlattice + bwdlattice
                    ## posteriors = np.exp(gamma.T - logsum(gamma, axis=1)).T
                    
                    # Calculate condition to stop inner loop
                    #     from pmhmm.m of [Gha97]
                    delmf = npsum(npsum(np.concatenate(posteriors.values(),axis=1)*\
                                        np.concatenate(logPost.values(),axis=1)))- \
                            npsum(npsum(np.concatenate(posteriors.values(),axis=1)*\
                                        np.concatenate(logPost0.values(),axis=1)))
                    ## print delmf#DEBUG
                    if delmf < nframes*0.000001 and inn>0:
                        itermf = inn
                        break
                
                # curr_logprob += lpr
                self._accumulate_sufficient_statistics_var_chain(
                    stats, seq, posteriors, fwdlattice,
                    bwdlattice, params)
                # logprob.append(curr_logprob)
            
            # Check for convergence.
            # TODO: compute true likelihood to take decision
            ## if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
            ##     break

            # correcting accumulated stats
            for n in range(self._n_chains):
                n0 = npsum(self.n_states[:n])
                n1 = n0 + self.n_states[n]
                #stats['cor'][np.ix_(idxStates,idxStates)]
                stats['cor'][n0:n1,n0:n1] = np.diag(stats[n]['post'])
                
            stats['cor'] = 0.5 * (stats['cor'] + stats['cor'].T)
            
            # at this point, links with [Gha97]:
            #     (especially for subclass GaussianFHMM)
            #     stats['obs'] -> GammaX
            #     stats['post'] -> gammasum
            #     stats['cor'] -> Eta
            #     stats['trans'] -> Xi
            
            # Maximization step
            self._do_mstep(stats, params, **kwargs)
            
        return self
    
    def _init(self, obs, params):
        if (hasattr(self, 'n_features')
            and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        
        self.n_features = obs[0].shape[1]
        
        for n in range(self._n_chains):
            self.HMM[n]._init(obs, params)

        self.startprob = None
        self.transmat = None
    
    @property
    def n_chains(self):
        """Number of states in the model."""
        return self._n_chains
    
    @property
    def n_states(self):
        """Number of states in the model."""
        return self._n_states
    @property
    def n_states_all(self):
        """Number of states in the model."""
        return self._n_states_all
    
    def _initialize_sufficient_statistics_var(self):
        stats = {}
        for n in range(self._n_chains):
            stats[n] = self.HMM[n]._initialize_sufficient_statistics()
        
        # specific to FHMM: correlation expectation
        #    [Gha97] : the sum_t < S_t S_t'>
        stats['cor'] = np.zeros([self._n_states_all,
                                 self._n_states_all])
        return stats
    
    def _accumulate_sufficient_statistics_var(self, stats, seq,
                                              frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """distribute the results and compute sufficient statistics
        for each chain
        
        
        """
        for n in range(self.n_chains):
            self.HMM[n]._accumulate_sufficient_statistics(\
                stats[n], seq, frameVarParams[n],
                posteriors[n], fwdlattice[n],
                bwdlattice[n], params)
        
        # specific to FHMM: first order correlation expectation
        #    [Gha97] : the sum_t < S_t S_t'>
        postTot = np.concatenate(posteriors.values(), axis=1)
        stats['cor'] += np.dot(postTot.T, postTot)
        del postTot
    
    def _accumulate_sufficient_statistics_var_chain(self, stats, seq,
                                                    posteriors, fwdlattice,
                                                    bwdlattice, params):
        """distribute the results and compute sufficient statistics
        for each chain
        
        
        """
        for n in range(self.n_chains):
            frameLogVarParams = self._compute_var_params_chain(seq,
                                                     posteriors=posteriors,
                                                     chain=n)
            self.HMM[n]._accumulate_sufficient_statistics(\
                stats[n], seq, frameLogVarParams,
                posteriors[n], fwdlattice[n],
                bwdlattice[n], params)
        
        # specific to FHMM: first order correlation expectation
        #    [Gha97] : the sum_t < S_t S_t'>
        postTot = np.concatenate(posteriors.values(), axis=1)
        stats['cor'] += np.dot(postTot.T, postTot)
        del postTot

    def _init_posterior_var(self, obs, method='random', **kwargs):
        """initialize the posterior probabilities with desired method:
            'random'
            'equi'
            'nmf'
            'lms'
            'triangle'
        """
        init_fun_dict = {'random'  : self._init_posterior_var_random,
                         'equi'    : self._init_posterior_var_equi,
                         'nmf'     : self._init_posterior_var_nmf,
                         'lms'     : self._init_posterior_var_lms,
                         'triangle': self._init_posterior_var_triangle}
        return init_fun_dict[method](obs=obs, **kwargs)
    
    def _init_posterior_var_random(self, obs, verbose=False, debug=False):
        """random posterior probabilities
        """
        nframes = obs.shape[0]
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains): 
            posteriors[n] = np.random.rand(nframes,
                                           self._n_states[n])
            posteriors[n] = normalize(posteriors[n], axis=1)
            logPost[n] = np.log(posteriors[n])
        return posteriors, logPost
    
    
    def _init_posterior_var_equi(self, obs, verbose=False, debug=False):
        """equiprobable posterior probabilities
        """
        nframes = obs.shape[0]
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains):
            posteriors[n] = np.ones([nframes, self._n_states[n]])/\
                            (1.*self._n_states[n])
            logPost[n] = np.log(posteriors[n])
        return posteriors, logPost
    
    
    def _init_posterior_var_nmf(self, obs, verbose=False, debug=False):
        """posteriors as normalized nmf amplitude coefficients,
        if it makes sense (for a given observation likelihood)
        """
        pass
    
    def _init_posterior_var_lms(self, obs, verbose=False, debug=False):
        """posteriors as normalized lms solution,
        if all components allowed everywhere, at once. 
        """
        pass
    
    def _init_posterior_var_triangle(self, obs, verbose=False, debug=False):
        """posteriors as normalized lms solution,
        if all components allowed everywhere, at once. 
        """
        nframes = obs.shape[0]
        posteriors = {}
        logPost = {}
        for n in range(self._n_chains):
            posteriors[n] = self._n_states[n]/2. - \
                            np.abs(np.arange(self._n_states[n])-\
                                   self._n_states[n]/2.) * \
                            np.vstack(np.ones([nframes,]))
            posteriors[n] /=  np.vstack(posteriors[n].sum(axis=1))
            logPost[n] = np.log(posteriors[n])
        return posteriors, logPost
    
    def _do_forward_pass_var(self, frameVarParams, **kwargs):
        
        logprob = {}
        fwdlattice = {}
        for n in range(self.n_chains):
            logprob[n], fwdlattice[n] = self.HMM[n]._do_forward_pass(\
                                                           frameVarParams[n],
                                                           **kwargs)
        
        return logprob, fwdlattice
    
    def _do_backward_pass_var(self, frameVarParams, **kwargs):
        
        bwdlattice = {}
        for n in range(self.n_chains):
            bwdlattice[n] = self.HMM[n]._do_backward_pass(frameVarParams[n],
                                                          **kwargs)
        return bwdlattice
    
    def _do_fwdbwd_pass_var_hmm_c(self, frameLogVarParams, chain):
        """n is the chain number
        """
        nframes = frameLogVarParams.shape[0]
        n = chain
        
        fwdbwd_callLine = "computeForwardBackward(nframes, n_states, "+\
                          "startprob, transmat,"+\
                          "dens, logdens, logAlpha, logBeta, alpha, beta);"
                
        if not frameLogVarParams.flags['C_CONTIGUOUS']:
            raise ValueError("FrameVarParams should be C contiguous!")
        fwdloglattice = np.zeros([nframes, self._n_states[n]])
        bwdloglattice = np.zeros([nframes, self._n_states[n]])
        fwdlattice = np.zeros([nframes, self._n_states[n]])
        bwdlattice = np.zeros([nframes, self._n_states[n]])
        inline(fwdbwd_callLine,
               arg_names=['nframes', 'n_states', \
                          'startprob', 'transmat', \
                          'dens', 'logdens', \
                          'logAlpha', 'logBeta', 'alpha', 'beta'],
               local_dict={'nframes': nframes,
                           'n_states': self._n_states[n],
                           'startprob': self.HMM[n].startprob,
                           'transmat': self.HMM[n].transmat,
                           'dens': np.array([0.]),# unused argument
                           'logdens': frameLogVarParams,
                           'logAlpha': fwdloglattice,
                           'logBeta': bwdloglattice,
                           'alpha': fwdlattice,
                           'beta': bwdlattice},
               support_code=fwdbwd_supportcode,
               extra_compile_args =['-O3 -fopenmp'], \
               extra_link_args=['-O3 -fopenmp'], \
               force=0, verbose=2)

        ##print frameLogVarParams #DEBUG
        ##print fwdlattice #DEBUG
        ##print fwdloglattice #DEBUG
        
        del fwdlattice
        del bwdlattice
        
        return fwdloglattice, bwdloglattice
    
    def _do_fwdbwd_pass_var_c(self, frameLogVarParams):
        nframes = frameLogVarParams[0].shape[0]
        
        fwdbwd_callLine = "computeForwardBackward(nframes, n_states, "+\
                          "startprob, transmat, "+\
                          "dens, logdens, logAlpha, logBeta, alpha, beta);"
        fwdloglattice = {}
        bwdloglattice = {}
        
        for n in range(self._n_chains):
            fwdloglattice[n], bwdloglattice[n] = \
                                           self._do_fwdbwd_pass_var_hmm_c(\
                                                    frameLogVarParams[n], n)
        
        return fwdloglattice, bwdloglattice
    
    def _compute_var_params(self, obs, posteriors):
        pass
    
    def _compute_var_params_chain(self, obs, posteriors, chain, debug=False):
        pass
    
    def _compute_log_likelihood(self, obs):
        pass
    
    def _do_mstep(self,stats,params,**kwargs):
        for n in range(self._n_chains):
            ## print params # DEBUG
            self.HMM[n]._do_mstep(stats[n], params, **kwargs)

class GaussianFHMM(_BaseFHMM):
    """Class implementing factorial HMM
    
    
    
    using the HMM base class provided by scikits.learn.hmm
    """

    def __init__(self, cvtype='full',
                 means_prior=None, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 n_states=[2,2],
                 startprob=None, transmat=None,
                 startprob_prior=None,
                 transmat_prior=None,
                 HMM=None):
        """
        """
        super(GaussianFHMM, self).__init__(n_states, 
                                           startprob, transmat,
                                           startprob_prior,
                                           transmat_prior)
        
        if startprob is None or \
               len(startprob)!=self._n_chains:
            startprob = [None] * self._n_chains
            
        if transmat is None or \
               len(transmat)!=self._n_chains:
            transmat = [None] * self._n_chains
        
        if startprob_prior is None or \
               len(startprob_prior)!=self._n_chains:
            startprob_prior = [None] * self._n_chains
        
        if transmat_prior is None or \
               len(transmat_prior)!=self._n_chains:
            transmat_prior = [None] * self._n_chains
        
        
        # self.HMM[n] should actually be a GaussianHMM, for this model
        # because it is characterized by its mean
        # That's gonna make the method _accumulate_sufficient_statistics
        # keep the running mean for each state of each chain
        self.HMM = {}
        for n in range(self._n_chains):
            self.HMM[n] = GaussianHMM(n_states[n], cvtype, startprob[n],
                 transmat[n], startprob_prior[n], transmat_prior[n],
                 means_prior, means_weight,
                 covars_prior, covars_weight)
        
        self._cvtype = cvtype
        if not cvtype in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad cvtype')
        
        self.means_prior = means_prior
        self.means_weight = means_weight
        
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
    
    def decode_var(self, obs, n_innerLoop=10, maxrank=None,
                   beamlogprob=-np.Inf, verbose=False, debug=False,
                   n_repeatPerChain=None, **kwargs):
        """decode_var
        
        returns most likely output mean, the estimated seq of states, and
        the posterior probabilities.
        
        For now, the sequence of states is calculated through the
        variational approximation, i.e. we first compute the variational
        parameters and the corresponding posteriors for each chain, iterate
        over that process n_innerLoop times, and at last outputs the states
        computed through Viterbi for each chain, on the log variational
        parameters as <<log likelihood>> (as suggested in [Gha97]).
        
        At last, we compute the mean corresponding to those sequences.
        
        Important: this is not the true sequence (should run Viterbi on all
        states, with true likelihood). It is possible to do, but very expensive
        in terms of memory (especially for source/filter subclass).
        
        """
        obs = np.asanyarray(obs)
        
        nframes = obs.shape[0]
        
        state_sequences, posteriors = super(GaussianFHMM, self).decode_var(\
                                             obs=obs,
                                             n_innerLoop=n_innerLoop,
                                             maxrank=maxrank,
                                             beamlogprob=beamlogprob,
                                             verbose=verbose,
                                             debug=debug,
                                             n_repeatPerChain=n_repeatPerChain,
                                             **kwargs)
        
        # compute outputmeans :
        outputMean = np.zeros([nframes,self.n_features])
        for n in range(self._n_chains):
            outputMean += self.means[n][state_sequences[n]]
        
        return outputMean, state_sequences, posteriors
    
    def _init(self, obs, params='stmc'):
        super(GaussianFHMM, self)._init(obs, params=params)
        
        if (hasattr(self, 'n_features')
            and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        
        self.n_features = obs[0].shape[1]
        
        # Initialize like [Gha97]
        if 'c' in params:
            cv = np.cov(np.concatenate(obs),rowvar=0)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars = _distribute_covar_matrix_to_match_cvtype(
                cv, self._cvtype, 1)
        
        if 'm' in params:
            self._means = {}
            for n in range(self._n_chains):
                # TODO: if cvtype not full, then probably issue here!
                self._means[n] = np.dot(np.random.randn(self._n_states[n],
                                                        self.n_features), \
                                   np.double(linalg.sqrtm(self._covars[0]))) / \
                                     self._n_chains + \
                                 np.concatenate(obs).mean(axis=0) / \
                                     self._n_chains
                
        ## print self.means#DEBUG
        ## print self.HMM#DEBUG
        ## print self.covars#DEBUG
    
    @property
    def cvtype(self):
        """Covariance type of the model.
        
        Must be one of 'spherical', 'tied', 'diag', 'full'.
        """
        return self._cvtype
    
    def _get_means(self):
        """Mean parameters for each state."""
        return self._means
    
    def _set_means(self, means):
        """setting the means of each chain: must be list of len==_n_chains
        """
        if len(means)!=self._n_chains:
            raise ValueError('means must be a list of length n_chains')
        
        for n in range(self._n_chains):
            if hasattr(self, 'n_features') and \
                   means[n].shape != (self._n_states[n], self.n_features):
                raise ValueError('means must have shape (n_states, n_features)')
            if means[n].shape[0] != self._n_states[n]:
                raise ValueError('means must have shape (n_states, n_features)')
        
        self._means = list(means)
        self.n_features = self._means[0].shape[1]
    
    means = property(_get_means, _set_means)
    
    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.cvtype == 'full':
            return self._covars
        elif self.cvtype == 'diag':
            return [np.diag(cov) for cov in self._covars]
        elif self.cvtype == 'tied':
            return [self._covars] * self._n_states
        elif self.cvtype == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars]
    
    def _set_covars(self, covars):
        """[Gha97] : covar matrix unique for all possible states"""
        covars = np.asanyarray(covars)
        # [Gha97] model with only one covar matrix for observation likelihood:
        _validate_covars(covars, self._cvtype, 1, self.n_features)
        self._covars = covars.copy()
    
    covars = property(_get_covars, _set_covars)
    
    def _compute_var_params(self, obs, posteriors):
        
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        frameVarParams={}
        # actually frameLogVarParams !
        
        ##print self._covars[0]#DEBUG
        L = linalg.cholesky(self._covars[0], lower=True)
        for n in range(self._n_chains):
            cv_sol = linalg.solve_triangular(L, self._means[n].T, lower=True).T
            deltas = np.sum(cv_sol ** 2, axis=1) # shape=(_n_states[n])
            obs_sol = linalg.solve_triangular(L,
                                              (obs - obsHat + \
                                               sigHatPerChain[n]).T,
                                              lower=True)
            frameVarParams[n] = np.dot(obs_sol.T, cv_sol.T) - \
                                0.5 * deltas # shape=(nframes, _n_states[n])
        
        return frameVarParams
    
    def _compute_var_params_chain(self, obs, posteriors, chain, debug=False):
        """Computes the variational parameters, which can be interpreted as
        the conditional likelihood, but for chain number `chain`. 
        """
        if debug: print "       Computing the variational parameters."
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        ##print self._covars[0]#DEBUG
        n = chain
        # the following is the same for all HM chains: should factorize
        L = linalg.cholesky(self.covars[0], lower=True)
        
        cv_sol = linalg.solve_triangular(L, self._means[n].T, lower=True).T
        deltas = np.sum(cv_sol ** 2, axis=1) # shape=(_n_states[n])
        obsHatChain = obsHat - sigHatPerChain[n]
        if hasattr(self, 'noiseLevel'):
            obsHatChain = np.maximum(obsHatChain, self.noiseLevel)
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        
        return np.dot(obs_sol.T, cv_sol.T) - \
                         0.5 * deltas # shape=(nframes, _n_states[n])
    
    def _compute_log_likelihood(self, obs):
        # return lmvnpdf(obs, self._means, self._covars, self._cvtype)
        pass
    
    def _xH_invC_x_chol(self, X):
        """returns np.dot(np.dot(x_n.T, inv(covar)), x_n),
        using cholesky decomposition of covar
        
        code inspired from scikits.learn.mixture
        
        used to compute Delta^(m) [Gha97] or could be...
        """
        nframes, nfeats = X.shape
        if nfeats!=self.n_features:
            raise ValueError("number of features should be n_features")
        L = linalg.cholesky(self._covars[0])
        cv_sol = linalg.solve_triangular(L, X.T, lower=True).T
        
        return np.sum(cv_sol ** 2, axis=1)
    
    def _do_mstep(self, stats, params=string.letters,
                  paramsHMMs = 'st', **kwargs):
        # this is going to update the priors and transitions in self.HMM
        #     avoid that it updates mean and cov (because they re GaussianHMM)
        
        super(GaussianFHMM, self)._do_mstep(stats, paramsHMMs, **kwargs)
        
        # compute SVD instead of inverse of stats['cor'] <-> Eta
        stats['cor'] = 0.5*(stats['cor'].T+stats['cor'])

        if 'm' in params or 'c' in params:
            U,s,V = linalg.svd(stats['cor'])
            Si = np.zeros([self._n_states_all,
                           self._n_states_all])
            
            for n in range(self._n_states_all):
                if s[n] >= (s.size * linalg.norm(s) * 0.001):
                    Si[n,n] = 1./s[n]
            
            # just to be sure:
            # print stats['cor'], np.dot(np.dot(U,np.diag(s)),V)#DEBUG
            ## print V.T.conj(), Si, U.T.conj(), stats['cor']#DEBUG
            ## print np.dot(V.T.conj(),V) #DEBUG OK
            ## print np.dot(U.T.conj(),U) #DEBUG OK
            ## print s #DEBUG 
            ## print np.dot(np.dot(V.T.conj(),np.diag(1./s)),np.dot(U.T.conj(),stats['cor']))#DEBUG - for some reason, this is *not* identity... which is annoying... probably because one of the singular values is small ?
            
            # update the means:
            GammaX = np.concatenate([stats[n]['obs'] \
                                     for n in range(self._n_chains)])
            
            ## print Si, GammaX #DEBUG
            ## print V,s,S,U, GammaX #DEBUG
        
        if 'm' in params:
            newMean = np.dot(np.dot(V.T.conj(), Si), np.dot(U.T.conj(), GammaX))
            ## print newMean
            ## print newMean #DEBUG
            for n in range(self._n_chains):
                n0 = npsum(self.n_states[:n])
                n1 = n0 + self.n_states[n]
                self._means[n] = newMean[n0:n1]
        
        ## print self.means
        
        # update covariance:
        if 'c' in params:
            newCov = stats['obsobs'] / stats['ntotobs'] - \
                     np.dot(GammaX.T, newMean) / stats['ntotobs']
            self._covars = [0.5 * (newCov + newCov.T)]
        ## print newCov #DEBUG
        # should check if covars is still valid (check the det)
        
        # update transition:
        #    done in each HMM, in call of super() above.
    
    def _initialize_sufficient_statistics_var(self):
        stats = super(GaussianFHMM, \
                      self)._initialize_sufficient_statistics_var()
        stats['obsobs'] = np.zeros([self.n_features, self.n_features])
        stats['ntotobs'] = 0
        return stats
    
    def _accumulate_sufficient_statistics_var(self, stats, seq,
                                              frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """
        params: 'c' covars
                'm' means
                't' transition matrices (in self.HMM)
                's' start probabilities (note: in scikits.learn.hmm, it's only
                                         meaningful with many observed
                                         sequences)
                
        """
        
        super(GaussianFHMM, self)._accumulate_sufficient_statistics_var(\
                                              stats, seq, frameVarParams,
                                              posteriors, fwdlattice,
                                              bwdlattice, params)
        stats['obsobs'] += np.dot(seq.T, seq)
        stats['ntotobs'] += seq.shape[0]
    
    def _accumulate_sufficient_statistics_var_chain(self, stats, seq,
                                              posteriors, fwdlattice,
                                              bwdlattice, params):
        """
        params: 'c' covars
                'm' means
                't' transition matrices (in self.HMM)
                's' start probabilities (note: in scikits.learn.hmm, it's only
                                         meaningful with many observed
                                         sequences)
                
        """
        
        super(GaussianFHMM, self)._accumulate_sufficient_statistics_var_chain(\
                                              stats, seq, 
                                              posteriors, fwdlattice,
                                              bwdlattice, params)
        stats['obsobs'] += np.dot(seq.T, seq)
        stats['ntotobs'] += seq.shape[0]

    def _init_posterior_var_lms(self, obs, verbose=False, debug=False):
        """least mean squares solution of obs = means*posteriors
        """
        eps = 1e-6
        nframes = obs.shape[0]
        means = np.concatenate(self.means.values()).T
        # h = np.zeros([nframes, means.shape[1]])
        # for n in range(nframes):
        #    if verbose:
        #        print 'frame',n
        #    h[n] = nnls(means, obs[n])[0]
        h = linalg.lstsq(means, obs.T)[0].T
        h = np.maximum(h, eps)
        posteriors = {}
        logPost = {}
        for n in range(self.n_chains):
            n0 = npsum(self.n_states[:n])
            n1 = n0 + self.n_states[n]
            # uncomment following if too much noise estimated
            ##if self.withNoiseF0 or self.withFlatFilter:
            ##    #do not take last element, which is noise
            ##    h[:, n1-1] = 0
            posteriors[n] = normalize(h[:,n0:n1], axis=1)
            logPost[n] = np.log(posteriors[n])
        
        return posteriors, logPost
        
    def _init_posterior_var_nmf(self, obs, verbose=False, debug=False):
        """posteriors as normalized nmf amplitude coefficients,
        if it makes sense (for a given observation likelihood)
        """
        
        nmfIter = 100
        nmfEps = 1e-4
        nmfMin = obs.min()
        nuObs = np.maximum(obs - nmfMin, nmfEps)
        
        nframes = obs.shape[0]
        means = np.concatenate(self.means.values())
        means = means - means.min() + nmfEps
        nmeans = means.shape[0]
        ##amplitudes = obs.max() / means.max() * \
        ##             (np.random.rand(nframes, nmeans))
        amplitudes = nuObs.max() / means.max() * \
                     np.ones((nframes, nmeans))
        amplitudes = np.maximum(amplitudes, nmfEps)
        
        for n in range(nmfIter):
            if verbose:
                print "iter",n,"over",nmfIter
            # NMF with IS rules:
            obsHat = np.dot(amplitudes, means)
            denom = np.dot(1. / np.maximum(obsHat, nmfEps), means.T)
            num = np.dot(nuObs / \
                         np.maximum(obsHat**2, nmfEps),
                         means.T)
            amplitudes = amplitudes * num / np.maximum(denom, nmfEps)
            amplitudes = np.maximum(amplitudes, nmfEps)
            if debug:
                plt.figure(200)
                plt.clf()
                plt.imshow(np.log(amplitudes), origin='lower',
                           interpolation='nearest')
                plt.colorbar()
                plt.draw()
                
        
        posteriors = {}
        logPost = {}
        
        for n in range(self._n_chains): 
            n0 = npsum(self.n_states[:n])
            n1 = n0 + self.n_states[n]
            posteriors[n] = normalize(amplitudes[:,n0:n1], axis=1)
            logPost[n] = np.log(posteriors[n])
        
        return posteriors, logPost


class SFFHMM(GaussianFHMM):
    """
    Source-Filter FHMM, based on source/filter model for
    speech, oriented towards tracking of frequencies of
    formants and F0.
    
    Each 
    """
    def __init__(self, cvtype='spherical',
                 means_prior=None, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 n_states=None,
                 startprob=None, transmat=None,
                 startprob_prior=None,
                 transmat_prior=None,
                 HMM=None, formantRanges=None, bwRange=None,
                 F0range=None, samplingRate=44100.,
                 stepNotes=4, chirpPerF0=1, NFT=2048,
                 withNoiseF0=False, withFlatFilter=True,
                 n_features=None,
                 withNoiseFloor=False):
        """Init method for Source/Filter FHMM class.
        
        Initializes the mean vectors for each Markov chain to
        F0/formant frequency Fp parameterized dictionary elements.
        
        The model is further extended thanks to an energy and filter
        components. In comparison with GaussianFHMM, this is necessary
        since we do not allow re-estimating the means.
        
        
        """
        
        self.samplingRate = samplingRate
        Fs = self.samplingRate
        self.NFT = NFT
        if n_features is None:
            self.n_features = self.NFT / 2 + 1
        else:
            self.n_features = n_features
        
        # creating spectrum bases:
        self._means = {}
        if not n_states is None:
            print "argument set for backward compatibility, but not used.."
        
        self._n_states = list()
        
        if F0range is None or len(F0range) != 2 or F0range[0] >= F0range[1]:
            self.F0range = [80,500]
        else:
            self.F0range = F0range
        
        minF0 = self.F0range[0]
        maxF0 = self.F0range[1]
        
        # TODO: generate WF0 with optional parameters
        #       notably Ot (and harmonize with generative method
        #                   soundSynthesis)
        self.F0Table, WF0 = \
            generate_WF0_chirped(minF0, maxF0, Fs, Nfft=self.NFT, \
                              stepNotes=stepNotes, \
                              lengthWindow=self.NFT, Ot=0.5, \
                              perF0=chirpPerF0, \
                              depthChirpInSemiTone=.15,
                              loadWF0=True,
                              analysisWindow='hanning')
        
        # adding some ground noise, so that all log will have same ground
        # level (20120111):
        #WF0 += ((WF0.max(axis=0))*1e-4*\
        #        np.random.randn(WF0.shape[0], WF0.shape[1]))**2
        # hard threshold instead:
        self.withNoiseFloor = withNoiseFloor
        if withNoiseFloor:
            self.noise0 = (WF0.max()) * 1e-4 # 0.00005 #needs this to reinitialize
            self.noise = self.noise0 #1e-10
            print "noise estimated", self.noise
        else:
            WF0 = np.maximum(WF0, (WF0.max(axis=0))*1e-8)
            
        self.withNoiseF0 = withNoiseF0
        if withNoiseF0:
            # print Fs, self.NFT, self.n_features #DEBUG
            # adding a -5dB/octave decaying noise 
            ##self.means[0] = np.vstack([np.log(WF0[:self.n_features].T),\
            ##                    np.log(WF0[:self.n_features].T).max() + \
            ##                    (- .5) * \
            ##                    (np.log2((50. + np.arange(self.n_features))/\
            ##                    (1. / Fs * self.NFT)) - \
            ##                    np.log2(50. * Fs / self.NFT))\
            ##                   ])#np.ones(self.n_features)])
            ##self.means[0] = np.vstack([np.log(WF0[:self.n_features].T),\
            ##                    np.log(WF0[:self.n_features]).max(axis=1)])
            self.means[0] = np.vstack([np.log(WF0[:self.n_features].T),\
                                np.log(WF0[:self.n_features]).mean() + \
                                (- 5.) * \
                                np.arange(self.n_features) / self.NFT])
            ##self.means[0] = np.vstack([np.log(WF0[:self.n_features].T),\
            ##                           np.log(WF0[:self.n_features]).mean() * \
            ##                           np.zeros(self.n_features)])
            self.F0Table = np.concatenate([self.F0Table, [0]])
            if withNoiseFloor:
                self.means[0] = np.log(np.exp(self.means[0]) + self.noise)
        else:
            if withNoiseFloor:
                self.means[0] = np.log(WF0[:self.n_features].T + self.noise)
            else:
                self.means[0] = np.log(WF0[:self.n_features].T)
        
        # normalizing WF0:
        # self.means[0] -= np.vstack(self.means[0].max(axis=1))
        self._n_states.append(self.means[0].shape[0])
        
        self.stepNotes = stepNotes
        self.chirpPerF0 = chirpPerF0

        self.maxFinFT = self.samplingRate / 2. 
        
        if formantRanges is None:
            ##self.formantRanges = {}
            ##self.formantRanges[0] = [50.0, 300.0] # glottal formant?
            ##self.formantRanges[1] = [550.0, 4000.0]
            ##self.formantRanges[2] = [1600.0, 4500.0]
            ##self.formantRanges[3] = [2400.0, 5000.0] 
            ##self.formantRanges[4] = [3500.0, 7000.0] 
            #self.formantRanges[5] = [4500.0, np.minimum(15000.0, self.maxFinFT)]
            ##self.formantRanges[5] = [4500.0, self.maxFinFT]
            # from durrieu/thiran ICASSP 2011 article:
            # schafer + hill (union), 20101019
            #self.formantRanges = {}
            #self.formantRanges[0] = [200.0, 1000.0]
            #self.formantRanges[1] = [550.0, 3100.0]
            #self.formantRanges[2] = [1700.0, 3800.0]
            #self.formantRanges[3] = [2400.0, 6000.0]
            #self.formantRanges[4] = [4500.0, np.minimum(8000.0, self.maxFinFT)]
            #1if self.maxFinFT>9000:
            #    self.formantRanges[5] = [ 6500., 15000.]
            #    self.formantRanges[6] = [ 8000., 20000.]
            #    self.formantRanges[7] = [15000., 22000.]
            # From TASLP 2013 article:
            formantsRange = {}
            formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
            formantsRange[1] = [ 550.0, 3500.0]
            formantsRange[2] = [1400.0, 4500.0]
            formantsRange[3] = [2400.0, 6000.0] # adding one for full band
            formantsRange[4] = [3300.0, 8000.0]
            formantsRange[5] = [4500.0, 8000.0]
            formantsRange[6] = [5500.0, 8000.0]
        else:
            self.formantRanges = {}
            for n in range(len(formantRanges)):
                self.formantRanges[n] = formantRanges[n]

        # TODO: pass these as arguments to __init__ 
        self.numberOfAmpsPerPole = 10
        self.numberOfFreqPerPole = 40
        self.bwRange, self.freqRanges, \
                      self.poleAmp, \
                      self.poleFrq, WGAMMA = genARbasis(\
            self.n_features, self.NFT, self.samplingRate, \
            maxF0=self.F0range[1]/4.,
            formantsRange=self.formantRanges,
            numberOfAmpsPerPole=self.numberOfAmpsPerPole,
            numberOfFreqPerPole=self.numberOfFreqPerPole)
        
        Fwgamma, Nwgamma = WGAMMA.shape
        self.numberOfFormants = self.freqRanges.shape[0]
        
        self.nElPerFor = Nwgamma / self.numberOfFormants
        self.withFlatFilter = withFlatFilter
        for p in range(self.numberOfFormants):
            if self.withFlatFilter:
                # adding an all 0 mean... good idea?
                self._means[p+1] = np.hstack([np.log(\
                                        WGAMMA[:,(p*self.nElPerFor):\
                                               ((p+1)*self.nElPerFor)]),
                                              np.atleast_2d(\
                    np.zeros(self.n_features)).T]).T
            else:
                # without that all 0 vector:
                self._means[p+1] = np.log(\
                                       WGAMMA[:,(p*self.nElPerFor):\
                                              ((p+1)*self.nElPerFor)]).T

            #self._means[p+1] = normalize(self._means[p+1], axis=1)
            # TODO: check whether normalization needed/interpretable:
            ##self._means[p+1] -= np.vstack(logsum(self._means[p+1], axis=1))
            self._n_states.append(self._means[p+1].shape[0])
        
        # feeding super constructor with the computed number of elements in bases
        # (i.e. n_states)
        super(SFFHMM, self).__init__(n_states=self._n_states, cvtype=cvtype,
                                     means_prior=means_prior,
                                     means_weight=means_weight,
                                     covars_prior=covars_prior,
                                     covars_weight=covars_weight,
                                     startprob=startprob,
                                     transmat=transmat,
                                     startprob_prior=startprob_prior,
                                     transmat_prior=transmat_prior)
        
        # self._covars = [np.array([[0.41108]])] # [np.array([[0.41108]])]
        # the following works for the log R^2 (log rayleigh just for
        # amplitudes)
        # 20120911T0800: why *4 ? covar should not depend on parameter, so
        #    no scaling effect (compared with rayleigh lambda=1...)
        self._covars = [np.array([[0.41108]])]# [4.*np.array([[0.41108]])] # [np.array([[0.41108]])]
        # from log-rayleigh study
        
        # self.generateTransMatForSFModel()
    
    def decode_var(self, obs, n_repeatPerChain=None,
                   thresholdEnergy=0., debug=False,
                   dataForInit=None, hopsizeForInit=256,
                   **kwargs):
        """decode_var
        
        runs super class decode_var with a reasonable default number of
        repeats of the inner loop per chain.
        
        Principle: this is to allow more loops for the source part, because
                   harder to initialize.
        """
        # estimating noise level, because under that noise level,
        # using gaussian (<=> euclidean distance) between logs is
        # irrelevant (because it amounts to big differences where
        # not perceptually important). Furthermore, present algo
        # seems sensitive to that inadequation.
        #
        # 20120105: no, dont do that: not working with audio signals -
        #     not as such anyway.
        if 'withNoiseFloor' in kwargs:
            self.withNoiseFloor = kwargs['withNoiseFloor']
            kwargs.pop('withNoiseFloor')
            
        if self.withNoiseFloor:
            self.means[0] = np.log(np.exp(self.means[0]) - \
                                   self.noise + \
                                   self.noise0)
            self.noise = self.noise0
            
        # this is a bit circumvoluted, but we need to initialize the
        # posteriors here, and the init function will raise a warning
        # when called from superclass decode_var.
        #
        # EDIT: this won't work, because posteriors and logposteriors
        #    are not global, so we need to make the time-domain data
        #    available as an attribute, and then delete it when it becomes
        #    unnecessary.
        self.data = dataForInit
        self.hopsize = hopsizeForInit
        #
        # a better handling of this is sought after...
        
        if thresholdEnergy!=0 and False:
            from matplotlib.mlab import find as mfind
            histVals, histBounds = np.histogram(obs, bins=200)
            histMeans = (histBounds[:-1]+histBounds[1:])/2.
            self.noiseLevel = histMeans[\
                mfind(np.cumsum(histVals*np.exp(histMeans)) / \
                      np.double((histVals*np.exp(histMeans)).sum()) < \
                      thresholdEnergy)[-1]]
            
            if debug:
                plt.figure(300)
                plt.clf()
                plt.imshow(np.maximum(obs, self.noiseLevel),
                           interpolation='nearest',
                           origin='lower')
                plt.title('observation, minimal is noise level '+\
                          str(self.noiseLevel))
                plt.colorbar()
                plt.draw()
            
            return super(SFFHMM, self).decode_var(obs=np.maximum(obs,
                                                             self.noiseLevel),
                                       n_repeatPerChain=n_repeatPerChain,
                                       debug=debug,
                                       **kwargs)
        else:
            return super(SFFHMM, self).decode_var(obs=obs,\
                                       n_repeatPerChain=n_repeatPerChain,
                                       debug=debug,
                                       **kwargs)
    
    def _init_posterior_var(self, obs, method='random', **kwargs):
        """initialize the posterior probabilities with desired method:
            'random'
            'equi'
            'nmf'
            'lms'
            'triangle'
            'lpc'
        """
        init_fun_dict = {'random'  : self._init_posterior_var_random,
                         'equi'    : self._init_posterior_var_equi,
                         'nmf'     : self._init_posterior_var_nmf,
                         'lms'     : self._init_posterior_var_lms,
                         'triangle': self._init_posterior_var_triangle,
                         'lpc'     : self._init_posterior_var_lpc}
        return init_fun_dict[method](obs=obs, **kwargs)
    
    def _init_posterior_var_lpc(self, obs, verbose=False,
                                debug=False):
        if not withLPC:
            raise NotImplementedError("LPC initialization is not possible."+
                                      " Install scikits.talkbox and retry.")
        if self.data is None:
            raise TypeError("For LPC computing, the time-domain data should "+
                            "also be provided to the program (keyword "+
                            "argument in decode_var: dataForInit).")
        # number of poles set by number of formants:
        lpcOrder = 2 * self.numberOfFormants + 1
        
        # guessing the hopsize from the dimensions of obs:
        nsamples = self.data.size
        nframes = obs.shape[0]
        # hopsize = int((nsamples - self.NFT) / (nframes - 1.))
        if verbose>1:
            print "hopsize", self.hopsize
        # LPC with talkbox:
        stLpc, rootLpc, freqLpc, sigmaS = lpcana.stlpc(
            longSignal=self.data, order=lpcOrder,
            windowLength=self.NFT,
            hopsize=self.hopsize)
        posteriors = {}
        logPost = {}
        
        # random init for F0 posteriors:
        posteriors[0] = np.random.rand(nframes,
                                       self._n_states[0])
        posteriors[0] = normalize(posteriors[0], axis=1)
        logPost[0] = np.log(posteriors[0])
        
        for n in range(self.numberOfFormants):
            p = n+1
            indInPoleFreq0 = (n*self.nElPerFor)
            indInPoleFreqE = ((n+1)*self.nElPerFor)
            posteriors[p] = np.ones([nframes,
                                     self._n_states[p]])
            #TODO FINISH IT HERE !# finish HERE !
            # getting the sequence of index in the formant dictionary
            #     first get the right formant sequence, in Hz: freqLpc[n]
            #     then find the closest frequencies in self.freqRanges[n]
            ##distHz = np.abs(np.vstack(self.) - freqLpc[n])
            
            # this for a pseudo-gaussian function around
            # the estimated LPC formant.
            sigmaDist = (1. * (self.freqRanges[n][1] # 4.
                               -self.freqRanges[n][0])
                         ) ** 2
            posteriors[p][:,:-1] = np.exp(
                - ((self.poleFrq[indInPoleFreq0:indInPoleFreqE])
                   - np.vstack(freqLpc[n])) ** 2
                / sigmaDist)
            posteriors[p] = normalize(posteriors[p], axis=1)
            logPost[p] = np.log(posteriors[p])
        
        return posteriors, logPost
    
    def _compute_energyComp(self, obs, posteriors, rfilter=None, debug=False):
        """_compute_energyComp
        
        given the current estimate of the posterior probabilities, for a given
        observation sequence, computes the energy parameter for each frame. 
        """
        
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        if debug:
            plt.figure(4)
            plt.clf()
            plt.subplot(211)
            #plt.imshow(((obs-obsHat).T)**2, origin='lower')
            plt.imshow(sigHatPerChain[0].T, origin='lower')
            plt.colorbar()
            plt.title("sigHatPerChain[0]")
            plt.draw()
        
        # compute energy per frame to compensate:
        #    the following should work, thanks to np's broadcasting
        L = linalg.cholesky(self.covars[0], lower=True)
        one_sol = linalg.solve_triangular(L,
                                          np.ones(self.n_features),
                                          lower=True)
        if rfilter is None:
            obsHatChain = obsHat
        else:
            obsHatChain = obsHat + rfilter
        ## if hasattr(self, 'noiseLevel'):
        ##     obsHatChain = np.maximum(obsHat, self.noiseLevel)
        ##     if debug:
        ##         print self.noiseLevel #DEBUG
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        logH = np.vstack(np.dot(one_sol.T, obs_sol))
        logH /= np.dot(one_sol,one_sol)
        
        return logH 
    
    def _compute_recording_condition(self, obs, posteriors, logH, debug=False):
        """Returns the log of the estimated frequency response for the
        recording condition filter.
        """
        obsHat = np.zeros_like(obs)
        for n in range(self._n_chains):
            obsHat += np.dot(posteriors[n], self._means[n])
            
        # 20120112
        # not rigorous: trying some smoothing on it, to see if it
        # works better
        rfilter = np.mean(obs - logH - obsHat, axis=0)
        # orderSmooth = self.NFT/10
        orderSmooth = 40
        rfilter = lfilter(np.ones([orderSmooth])/np.double(orderSmooth),
                          np.ones([1]), rfilter)
        
        return rfilter
    
    def _compute_var_params_chain(self, obs, posteriors, chain, debug=False,
                                  typeEnergy='oneForAll'):
        """_compute_var_params_chain
        
        to be able to select between 2 possible models, for the typeEnergy:
            'onePerF0' : we estimate one energy correction per F0. Somehow, as
                         of 2011/12/23, it would seem that these are too many
                         params to be estimated, and therefore leads to /bad/
                         results.
            'oneForAll': we estimate one energy correction per frame for all
                         the possible states. 2011/12/23: seems to work better
                         than the 'onePerF0' model. A reason could be that there
                         needs to be too many iterations otherwise. 
        """
        computeVarParamsFunc = {'onePerF0': self._compute_var_params_chain_f0,
                                'oneForAll': self._compute_var_params_chain_all}
        
        return computeVarParamsFunc[typeEnergy](obs=obs,
                                                posteriors=posteriors,
                                                chain=chain,
                                                debug=debug)
    
    def _compute_var_params_chain_all(self, obs, posteriors, chain,
                                      debug=False):
        """_compute_var_params_chain_all(self, obs, posteriors, chain,
                                         debug=False)
        """
        # We can compute the additional energy parameter before
        # computing the variational parameters,
        # and store that additional 
        #
        
        if debug:
            print "        Compute the contribution of the obs. to this chain."
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            obsHat += sigHatPerChain[n]
        
        # Calculating logH in a separate function
        ##if debug:
        ##    plt.figure(4)
        ##    plt.clf()
        ##    plt.subplot(211)
        ##    #plt.imshow(((obs-obsHat).T)**2, origin='lower')
        ##    plt.imshow(sigHatPerChain[0].T, origin='lower')
        ##    plt.colorbar()
        ##    plt.title("sigHatPerChain[0]")
        ##    plt.draw()
        ##
        ### compute energy per frame to compensate:
        ###    the following should work, thanks to np's broadcasting
        if debug:
            print "        Decomposing covar for efficient inversion."
        L = linalg.cholesky(self.covars[0], lower=True)
        if debug:
            print "        Re-estimating the energy and filters"
        logH = self._compute_energyComp(obs=obs, posteriors=posteriors,
                                        debug=debug)
        rfilter = self._compute_recording_condition(obs=obs,
                                                    posteriors=posteriors,
                                                    logH=logH,
                                                    debug=debug)
        obsHat += np.vstack(logH) + rfilter
        
        if debug:
            plt.figure(4)
            plt.subplot(212)
            plt.imshow(obsHat.T, origin='lower')
            plt.colorbar()
            plt.title("obshat with logh")
            plt.clim([obs.min(), obs.max()])
            plt.draw()
            
            plt.figure(5)
            plt.clf()
            plt.subplot(211)
            plt.plot(logH)
            plt.title("logH")
            plt.subplot(212)
            plt.plot(rfilter)
            plt.title("recording condition freq response")
            plt.draw()
        
        # this does not work with the noiseLevel thing: have to
        # copy the code here...
        ##return super(SFFHMM, self)._compute_var_params_chain(obs - \
        ##                                                 np.vstack(logH), \
        ##                                                posteriors=posteriors,
        ##                                               chain=chain,
        ##                                                debug=debug)
        
        if debug:
            print "        Computing the variational parameters"
        n = chain
        cv_sol = linalg.solve_triangular(L, self.means[n].T, lower=True)
        deltas = np.sum((cv_sol) ** 2, axis=0)
        obsHatChain = obsHat - sigHatPerChain[n]
        if hasattr(self, 'noiseLevel'):
            obsHatChain = np.maximum(obsHatChain, self.noiseLevel)
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        
        # if desired, update the mean for the source part,
        # with estimated noise floor:
        if self.withNoiseFloor:
            self._estimate_noise(posteriors=posteriors,
                                 obsm=(obs - np.vstack(logH) - rfilter),
                                 obsHat0=(obsHat - sigHatPerChain[0]),
                                 #step=0.001,
                                 debug=debug,
                                 verbose=True)
            
        if debug:
            print "        Matrix product to finish computation of var params"
        return np.dot(obs_sol.T, cv_sol) - \
               0.5 * deltas
    
    def _compute_var_params_chain_f0(self, obs, posteriors, chain, debug=False):
        """_compute_var_params_chain(self, obs, posteriors, chain, debug=False)
        """
        # 20111219T1645: that wont do, check notes about it
        #                Have to write the whole method completely.
        # super(SFFHMM, self)._compute_var_params(obs, posterior=posterior)
        # 20111220T1039: done?
        # We can compute the additional energy parameter before
        # computing the variational parameters,
        # and store that additional 
        
        obsHat = np.zeros_like(obs)
        sigHatPerChain = {}
        for n in range(self._n_chains):
            sigHatPerChain[n] = np.dot(posteriors[n], self._means[n])
            #print 'post [n] n', n, np.any(np.isnan(posteriors[n]))
            #print 'sigHatPerChain[n] n', n, \
            #      np.any(np.isnan(sigHatPerChain[n]))#DEBUG
            obsHat += sigHatPerChain[n]
            
        if debug:
            plt.figure(4)
            plt.clf()
            plt.subplot(211)
            #plt.imshow(((obs-obsHat).T)**2, origin='lower')
            plt.imshow(sigHatPerChain[0].T, origin='lower')
            plt.colorbar()
            plt.title("obshat without logh")
            plt.draw()
        
        L = linalg.cholesky(self.covars[0], lower=True)
        
        one_sol = linalg.solve_triangular(L,
                                          np.ones(self.n_features),
                                          lower=True)
        w0_sol = linalg.solve_triangular(L,
                                         self.means[0].T,
                                         lower=True)
        obsHatChain = obsHat - sigHatPerChain[0]
        if hasattr(self, 'noiseLevel'):
            obsHatChain = np.maximum(obsHatChain, self.noiseLevel)
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        # compute energy per frame to compensate:
        logH = np.vstack(np.dot(one_sol.T, obs_sol)) - \
               np.dot(w0_sol.T, one_sol)
        logH /= np.dot(one_sol,one_sol)
        
        sigHatPerChain[0] += np.vstack((logH * posteriors[0]).sum(axis=1))
        obsHat += np.vstack((logH * posteriors[0]).sum(axis=1))
        
        rfilter = self._compute_recording_condition(obs=obs,
                      posteriors=posteriors,
                      logH=np.vstack((logH * posteriors[0]).sum(axis=1)),
                      debug=debug)
        if debug:
            plt.figure(4)
            plt.subplot(212)
            plt.imshow(((obs-obsHat).T)**2, origin='lower')
            plt.colorbar()
            plt.title("obshat with logh")
            plt.draw()
            
            plt.figure(5)
            plt.clf()
            plt.imshow(logH.T, origin='lower')
            plt.colorbar()
            plt.title("logH")
            plt.draw()
            
            ## for n in range(self._n_chains):
            ##     plt.figure(10+n)
            ##     plt.clf()
            ##     plt.imshow(sigHatPerChain[n].T,
            ##                origin='lower', interpolation='nearest')
            ##     plt.colorbar()
            ##     plt.draw()
        
        ##print self._covars[0]#DEBUG
        n = chain
        # the following is the same for all HM chains: shoudl factorize
        
        cv_sol = linalg.solve_triangular(L, self.means[n].T, lower=True)
        deltas = np.sum((cv_sol) ** 2, axis=0) # shape=(_n_states[n])
        obsHatChain = obsHat - sigHatPerChain[n]
        if hasattr(self, 'noiseLevel'):
            obsHatChain = np.maximum(obsHatChain, self.noiseLevel)
        obs_sol = linalg.solve_triangular(L,
                                          (obs - obsHatChain).T,
                                          lower=True)
        
        # estimate noise
        if self.withNoiseFloor:
            self._estimate_noise(
                posteriors=posteriors,
                obsm=(obs - \
                      np.vstack((logH * posteriors[0]).sum(axis=1)) - \
                      rfilter),
                obsHat0=(obsHat - sigHatPerChain[0]),
                #step=0.001,
                debug=debug)
        if chain==0:
            # compensation of energy for source part:
            enLLobs = np.vstack(np.dot(obs_sol.T, one_sol)) * logH
            # compensation of the energy in the deltas:
            halfdeltasn = 0.5 * np.dot(one_sol, one_sol) * (logH**2) + \
                          np.dot(one_sol, cv_sol) * logH
            ## shape = (nframes,)
            del one_sol
            return np.dot(obs_sol.T, cv_sol) + enLLobs - \
                       0.5 * deltas - halfdeltasn#shape=(nframes,_n_states[n])
        else:
            return np.dot(obs_sol.T, cv_sol) - \
                       0.5 * deltas # shape=(nframes, _n_states[n])

    def _estimate_noise(self, posteriors, obsm, obsHat0,
                        step=1., debug=False, verbose=False):
        """Estimates the noise floor to be added to each mean,
        using newton's gradient method.
        
        heavily depends on the variables computed from
        methods _compute_var_params_chain_* 
        One should therefore not really use it directly (the computation
        is done in the aforementioned methods, provided
        withNoiseFloor is True).
        
        obsm should be (obs - np.vstack(logH) - rfilter)
        obsHat0 should be (obsHat - sigHatPerChain[0])
        
        ### the commonNoise option has been removed for now ###
        ### technical too complicated / expensive to put in operation ### 
        if commonNoise, then self.noise is a scalar, otherwise,
        self.noise is an array the number of signal frames.
        
        Personal note: cf. calculations at 20120306
        """
        ##############################
        # TODO: test noise estimation
        ##############################
        if debug:
            print "            For debug, trying not to estimate noise."
        return
        ##############################
        
        if verbose:
            print "        Noise estimation"
        eps = 1e-30
        # "posterior mean" for exponential of W0:
        factor1 = np.dot(posteriors[0],
                         1 / np.maximum(np.exp(self.means[0]), eps))
        # factor1 = np.maximum(factor1, eps)
        
        expectedOccState = np.sum(posteriors[0], axis=0)
        
        # update noise:
        #    Gradient and Hessian, for Newton gradient ascent:
        gradient = np.sum(obsHat0 * factor1) + \
                   np.dot(expectedOccState,
                          np.sum(self.means[0] / \
                                 np.exp(self.means[0]),
                                 axis=1)) - \
                   np.sum(obsm * factor1)
        
        # remove previous value, before updating noise value:
        self.means[0] = np.log(np.maximum(np.exp(self.means[0]) - \
                                          self.noise, eps))
        
        stepMu = - step * 1e-10 # ad-hoc gradient step, hessian too chaotic
        if stepMu < 0:
            # we are looking for a maximum, so hessian should be < 0
            # and if hessian too close to 0, that means that the optimization
            # is somehow not reliable (the log proba attains a "plateau")
            self.noise -= stepMu * gradient
        self.noise = np.maximum(self.noise, self.noise0)
        # limiting the noise to a certain fraction of the original
        # energy of the means
        self.noise = np.minimum(self.noise, 1e-4*self.noise0)
        # 0.005*np.exp(self.means[0].max()))
        ##if debug:
        ##    print "For debug purposes, trying not to estimate."
        ##self.noise = self.noise0 # DEBUG
        if verbose:
            print "        Noise estimated", self.noise, gradient # , hessian
        self.means[0] = np.log(np.exp(self.means[0]) + self.noise)        
    
    def generateTransMatForSFModel(self,
                                   stepSize, # s
                                   ):
        """generateTransMatForSFModel(stepSize)
        
        stepSize should be expressed in seconds, and corresponds to
        the step size between two analysis frames
        
        generates default transition matrices for the HMMs
        """
        # for parameters learnt from hillenbrand:
        stepHill = 0.026 # s, on average
        ##note : on Deng's DB:
        # stepSize = 0.01 #s
        # alphaFp = 0.038 #...
        ##according to formula below, we should have expected 0.08 for alphaFp!
        ##it is therefore not sure how these statistics could be
        ##made more robust. Dependent on phonetic class?
        
        # Normal distribution, learnt on Hillenbrand
        ##alphaF0 = 0.122877874162 #1.
        # seems that playin with this enables/disables noise...
        ##alphaFp = .01 #learnt from hillenbrand
        # Laplace distribution, learnt from hillenbrand
        # seems that playin with this enables/disables noise...
        alphaF0 = 0.278940139363 * stepHill / stepSize # 1. 
        alphaFp = 100. * 0.0308606370504 * stepHill / stepSize # .01 
        
        # HMM for the source part (n=0)
        n = 0
        ##logtransmatVec = np.arange(self._n_states[n]) / self.chirpPerF0  / \
        ##                 np.double(self.stepNotes)
        if self.withNoiseF0:
            logtransmatVec = np.log2(np.hstack([np.vstack(self.F0Table[:-1])]*\
                                               self.chirpPerF0)).flatten()
            logtransmatVec = np.concatenate([logtransmatVec, [0.]])
        else:
            logtransmatVec = np.log2(np.hstack([np.vstack(self.F0Table)]*\
                                               self.chirpPerF0))
        # Normal distribution, function of squared difference
        ##transmat = np.exp(- (np.vstack(logtransmatVec) - \
        ##                     logtransmatVec) ** 2 / alphaF0)
        # Laplace distribution, function of absolute difference
        
        transmat = np.exp(- np.abs(np.vstack(logtransmatVec) - \
                                   logtransmatVec) / alphaF0)
        # transition from voiced to unvoiced
        transFto0 = transmat[0,0]#self.stepNotes*self.chirpPerF0]#,0]
        #,self.stepNotes]
        # transition from unvoiced to voiced
        trans0toF = transmat[0,0]#,self.stepNotes]
        # transition from unvoiced to unvoiced
        trans0to0 = transmat[0,0]#self.stepNotes*self.chirpPerF0]#,0]
        #self.stepNotes]
        # thresholding, so that transition still possible to other pitches
        # transmat = np.maximum(transmat, transmat[0,3*self.stepNotes])
        ### zeroing stuff: hard smoothing !
        transmat[transmat<transmat[0,5*self.stepNotes*self.chirpPerF0]] = 0.
        
        if self.withNoiseF0:
            transmat[:,-1] = transFto0
            transmat[-1] = trans0toF
            transmat[-1,-1] = trans0to0
        
        self.HMM[n].transmat = normalize(transmat, axis=1)
        
        for n in range(1,self._n_chains):
            nElSameFreq = np.int(self.nElPerFor / \
                                 np.double(self.freqRanges[n-1].size))
            idx1 = (n-1) * self.nElPerFor
            idx2 = ( n ) * self.nElPerFor
            # logtransmatVec = np.arange(self.nElPerFor)
            logtransmatVec = np.log2(self.poleFrq[idx1:idx2])
            if self.withFlatFilter:
                logtransmatVec = np.concatenate([logtransmatVec, \
                                                 [-self.nElPerFor]])
            # normal transitions:
            ##transmat = np.exp(- (np.vstack(logtransmatVec) - \
            ##                     logtransmatVec)**2 / alphaFp)
            # laplacian transitions:
            transmat = np.exp(- np.abs(np.vstack(logtransmatVec) - \
                                 logtransmatVec) / alphaFp)
            # thresholding, so that transition still possible to other
            # formant freqs
            # transition from voiced to unvoiced
            transFto0 = transmat[0,nElSameFreq]
            # transition from unvoiced to voiced
            trans0toF = transmat[0,nElSameFreq]
            # transition from unvoiced to unvoiced
            trans0to0 = transmat[0,nElSameFreq]
            #transmat = np.maximum(transmat, transmat[0, *nElSameFreq])

            # 20120112
            ### zeroing stuff: hard smoothing !
            transmat[transmat<transmat[self.nElPerFor/2,
                                   self.nElPerFor/2+self.nElPerFor/4]] = 0.#2
            
            if self.withFlatFilter:
                transmat[:,-1] = transFto0
                transmat[-1] = trans0toF
                transmat[-1,-1] = trans0to0
            # print n, self._n_states[n], transmat.shape #DEBUG
            self.HMM[n].transmat = normalize(transmat, axis=1)
            
        # normalizing so as not to give too much weight to extremes:
        #     operates on _log_transmat to bypass "well-formedness" of transmat
        n = 0
        if self.withNoiseF0:
            self.HMM[n]._log_transmat[:-1,:-1] -= \
                np.vstack(self.HMM[n]._log_transmat[:-1,:-1].max(axis=1))
            self.HMM[n]._log_transmat[-1] = \
                self.HMM[n]._log_transmat[:-1,:-1].max() / 10.
            # - np.log(self._n_states[0])
            self.HMM[n]._log_transmat[:,-1] = self.HMM[n]._log_transmat[-1] 
        else:
            self.HMM[n]._log_transmat -= \
                np.vstack(self.HMM[n]._log_transmat.max(axis=1))
        for n in range(1, self.n_chains):
            if self.withFlatFilter and False:
                self.HMM[n]._log_transmat[:-1,:-1] -= \
                    np.vstack(self.HMM[n]._log_transmat[:-1,:-1].max(axis=1))
            else:
                self.HMM[n]._log_transmat -= \
                    np.vstack(self.HMM[n]._log_transmat.max(axis=1))
                
    def soundSynthesisParams(self, logH, rfilter, states,
                             hopsize, winsize, debug=False):
        """soundSynthesis
        Given the parameters, this function creates a sound PCM array,
        coresponding to the provided states and posteriors,
        as output by self.decode_var(obs), along with the corresponding
        energy parameter logH and recording condition rfilter.
        """
        n_chains = len(states)
        if n_chains != self._n_chains:
            raise AttributeError("Not the right number of chains in states.")
        
        for n in range(self._n_chains):
            if states[n].max() > self._n_states[n]:
                raise AttributeError("value in states too big.")
        
        odgd = generate_ODGD_vec(F0vec=self.F0Table[states[0]/self.chirpPerF0],
                                 Fs=self.samplingRate,
                                 lengthOdgd=hopsize,
                                 Ot=0.5, 
                                 t0=0.0,
                                 analysisWindowType='hanning',
                                 noiseLevel=1.)
        
        ODGD, F, N = stft(data=odgd,
                          window=hann(winsize),
                          hopsize=hopsize,
                          nfft=self.NFT, \
                          fs=self.samplingRate)
        # ODGD should be at least same length as obs:
        ODGD = ODGD[:self.n_features]
        # ODGD /= np.vstack(np.abs(ODGD).max(axis=1))
        # OH! check the above, this looks bad... should be:
        ODGD /= np.abs(ODGD).max(axis=0)
        # TODO: also check that the unvoiced parts have the
        # right "relative" energy as before
        # (check __init__ how means[0][-1] was initialized)
        ODGD = np.concatenate((ODGD.T,
                               np.zeros([states[0].size-ODGD.shape[1],
                                         self.n_features])))
        # Making a filter TF mask:
        # the means are, according to the article, the logarithm
        # of the power spectra (PSD or power spectrum).
        # in order to give the estimate of the corresponding Fourier
        # transform, we therefore have to take the exponential of half of
        # the mean... 
        S = np.exp(np.sum([0.5*self.means[n][states[n]] \
                           for n in range(1, self.n_chains)],
                          axis=0) +
                   0.5*logH + 0.5*rfilter)
        if self.withNoiseF0:
            # DEBUG
            # print (states[0]==(self._n_states[0]-1)).sum(), states[0].shape
            if debug:
                plt.figure()
                plt.subplot(211)
                plt.imshow(np.log(S).T)
                loclim, hiclim = plt.gca().get_images()[0].get_clim()
                plt.colorbar()
            # multiply by the noise profile given in self.means[0][-1]
            # and correct amplitude (because noise level = 1.),
            # correction by np.sqrt(self.NFT), since that's the
            # constant that multiplies the FT, normally.
            # (typically, an std of 1 gives an 2048-pts FT with values
            # np.sqrt(2048))
            S[states[0]==(self._n_states[0]-1)] = \
                S[states[0]==(self._n_states[0]-1)] * \
                np.exp(0.5 * self.means[0][-1]) / np.sqrt(self.NFT)
            if debug:
                plt.subplot(212)
                plt.imshow(np.log(S).T)
                plt.clim([loclim,hiclim])
                plt.colorbar()
                plt.draw()
        
        S = ODGD * S
        
        # S *= np.exp(0.5*logH) # without the "recording condition"
        
        synth = istft(X=S.T,
                      window=hann(winsize),
                      hopsize=np.double(hopsize),
                      nfft=self.NFT)
        
        return synth
        
    def soundSynthesis(self, obs, states, posteriors, hopsize, winsize,
                       debug=False):
        """soundSynthesis
        Given the parameters, this function creates a sound PCM array,
        coresponding to the provided obs, states and posteriors,
        as output by self.decode_var(obs)
        """
        n_chains = len(states)
        if n_chains != self._n_chains:
            raise AttributeError("Not the right number of chains in states.")
        
        for n in range(self._n_chains):
            if states[n].max() > self._n_states[n]:
                raise AttributeError("value in states too big.")
        
        logH = self._compute_energyComp(obs=obs,
                                        posteriors=posteriors,
                                        debug=False)
        rfilter = self._compute_recording_condition(obs=obs,
                                                    posteriors=posteriors,
                                                    logH=logH,
                                                    debug=False)
        
        return self.soundSynthesisParams(logH=logH, rfilter=rfilter,
                                         states=states,
                                         hopsize=hopsize, winsize=winsize,
                                         debug=debug)

######################################
# FOLLOWING ARE ADDITIONAL FUNCTIONS:
######################################

# a useful function for Fourier transforms:
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

# useful methods, to create bases of spectra
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
            ## print "making some chirped elements..."
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
    
    # normalization added so that less noise after in estimation
    odgd /= np.abs(odgd).max() # 20130516 DJL removed and replaced by :
    ##odgd /= np.double(F0) # 20130516 DJL same as generate_ODGD_vec
    #20130516 DJL: reverting to normalize by max, to fit article
    
    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
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
    partialMax = np.floor((Fs / 2) / np.maximum(F1, F2))
    
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
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)
    
    return odgd, odgdSpectrum

def genARbasis(numberFrequencyBins, sizeOfFourier, Fs, \
               formantsRange=None, \
               bwRange=None, \
               numberOfAmpsPerPole=5, \
               numberOfFreqPerPole=60, \
               maxF0 = 1000.0):
    """genARbasis: create a basis for given formant ranges
    
    creates an AR-1 spectra basis, one per provided formant range
    
    """
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
                WGAMMA[:,elementNb] = 1 / \
                   np.abs(1 - \
                          2.0 * \
                          poleAmp[elementNb] * \
                          np.cos(2.0 * np.pi * poleFrq[elementNb] / \
                                 np.double(Fs)) * cplxExp +
                          (poleAmp[elementNb] * cplxExp) ** 2\
                          ) ** 2
    
    return bwRange, freqRanges, poleAmp, poleFrq, WGAMMA

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

def stft(data, window=hann(2048), hopsize=256, nfft=2048, \
         fs=44100):
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
    dataInput = np.concatenate((np.zeros(lengthWindow / 2.0),data))          
    lengthData = dataInput.size
    
    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = np.ceil((lengthData - lengthWindow) / np.double(hopsize) \
                           + 1) + 1  
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    dataInput = np.concatenate((dataInput,
                                np.zeros([newLengthData - lengthData])))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2.0 + 1
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * dataInput[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, nfft);
    
    F = np.arange(numberFrequencies) / np.double(nfft) * fs
    N = np.arange(numberFrames) * hopsize / np.double(fs)
    
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

