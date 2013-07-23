"""test_sffhmm

Example file for the use of the SFFHMM class.
Modify the name of the file (a WAV file) that you desire to process.

Please refer to the article for further details on the theory behind
the algorithm:
Jean-Louis Durrieu and Jean-Philippe Thiran,
`Source/Filter Factorial Hidden Markov Model, 
with Application to Pitch and Formant Tracking`, 
submitted in 2013.

Jean-Louis Durrieu, 2013

"""

import sffhmm as fhmm
import scipy.io.wavfile as wav
import numpy as np
import os

import matplotlib.pyplot as plt

# The name of the file to process:
filename = os.environ['HOME'] + '/work/BDD/formants/presentingSNNDSPSFFT.wav'

# algorithm parameters, resp.:
#   window length
#   stepsize for the time-freq representation
#   size of Fourier transform
#   number of chirps per F0
#   number of iterations per inner loop
#   integrate a noise floor or not
# 
lenWin = 2048
steWin = 256
NFT = 2048
chirpPerF0 = 1
ninner = 3
withNoiseFloor = True

# read the audio data, normalizing and changing the data type:
fs, data = wav.read(filename)
data = np.double(data)
data /= (abs(data)).max()

# time-frequency representation: 
obs, freqs, timeStamps = fhmm.stft(data, fs=fs, window=fhmm.hann(lenWin),
                                   hopsize=steWin, nfft=NFT)
obs2 = (np.array(abs(obs.T)**2, order='C'))
obs2 = np.log(obs2)

# working only on the first frequency bins
F = np.int32(8000. / fs * NFT)
obs2 = obs2[:,:F]

# displaying the spectrogram
plt.figure(10)
plt.clf()
plt.subplot(111)
plt.imshow(obs2.T, origin='lower', interpolation='nearest')
plt.title('observation')
plt.colorbar()
plt.draw()

# the formant ranges for a sampling rate of 16kHz
formantsRange = {}
formantsRange[0]  = [  200.0, 1500.0] # check hillenbrand data
formantsRange[1]  = [  550.0, 3500.0]
formantsRange[2]  = [ 1400.0, 4500.0]
formantsRange[3]  = [ 2400.0, 6000.0] # adding one for full band
formantsRange[4]  = [ 3300.0, 7000.0]
formantsRange[5]  = [ 4500.0, 8000.0]
formantsRange[6]  = [ 5500.0, 8000.0]
# when allowing higher sampling rates, elevating the number
# of formants to track as well as their frequency range might be needed:
#formantsRange[7]  = [ 6500.0,12000.0]
#formantsRange[8]  = [ 8000.0,15000.0]
#formantsRange[8]  = [10000.0,20000.0]
#formantsRange[9] = [15000.0,22000.0]

# instantiating the SFFHMM model:
sffhmm = fhmm.SFFHMM(samplingRate=fs,NFT=NFT,withNoiseF0=True,
                     withFlatFilter=True,
                     formantRanges=formantsRange,
                     F0range=[100,500],
                     chirpPerF0=chirpPerF0,stepNotes=16,
                     n_features=F, withNoiseFloor=withNoiseFloor)
# this generates the transition probabilities for the SFFHMM model:
sffhmm.generateTransMatForSFModel(stepSize=steWin/np.double(fs))

# decode_var runs the EM algorithm on the provided observation
# spectrogram. As detailed in the article, this algorithm computes
# the posterior probabilities from the proposal likelihood, and the parameters
# encoding the spectrum.
#
#     mpost
#       the posterior mean given the parameters and under the proposed
#       variational likelihood.
#     states
#       the state sequence from the Viterbi decoding
#     posteriors
#       the posterior probabilities under the variational likelihood
#
mpost, states, posteriors = sffhmm.decode_var(obs=obs2, n_innerLoop=ninner,
                                              verbose=True, debug=False,#True,
                                              thresholdEnergy=0.,
                                              #thresholdEnergy=0.00000001,
                                              postInitMeth='triangle',#'lpc', #'triangle',#
                                              withNoiseFloor=withNoiseFloor,
                                              hopsizeForInit=steWin,
                                              dataForInit=data)

# To resynthesize the signal, we need the amplitude ...
logH = sffhmm._compute_energyComp(obs=obs2,
                                  posteriors=posteriors,
                                  debug=True)
# ... and the recording condition:
rfilter = sffhmm._compute_recording_condition(obs=obs2,
                                              posteriors=posteriors,
                                              logH=logH,
                                              debug=True)

mpost += logH + rfilter

# This converts the states in "index in the dictionaries" into
# states in "index of the frequency bin"
statesInNF = {}
statesInNF[0] = sffhmm.F0Table[states[0]/chirpPerF0] / np.double(fs) * NFT
for n in range(sffhmm.numberOfFormants):
    idx1 = (n) * sffhmm.nElPerFor
    idx2 = (n+1) * sffhmm.nElPerFor
    freqs = np.concatenate([sffhmm.poleFrq[idx1:idx2], [0]])
    statesInNF[n+1] = freqs[states[n+1]] / \
                      np.double(fs) * NFT

# drawing the spectrogram and the estimated formants
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.imshow(obs2.T,interpolation='nearest',origin='lower')
ax1.set_title('observation')
for n in range(sffhmm.numberOfFormants+1):
    ax1.plot(statesInNF[n], '.-')
ax1.axis('tight')
ax2=fig.add_subplot(212, sharex=ax1,sharey=ax1)
ax2.imshow(mpost.T,interpolation='nearest',origin='lower')
ax2.set_title('posterior mean')
for n in range(sffhmm.numberOfFormants+1):
    ax2.plot(statesInNF[n], '.-')
ax2.get_images()[0].set_clim(ax1.get_images()[0].get_clim())
#plt.colorbar()
ax2.axis('tight')
fig.canvas.draw()

# saving the parameters needed for the sound synthesis 
import datetime
dateNow = datetime.datetime.strftime(\
        datetime.datetime.now(), format='%Y%m%dT%H%M')

savefilename = 'moi_%s.npz'%dateNow
np.savez(savefilename,
         logH=logH,
         rfilter=rfilter,
         states=[np.uint16(states[n]) \
                 for n in range(len(states))],
         steWin=steWin, lenWin=lenWin)

# sound synthesis
synth = sffhmm.soundSynthesisParams(logH=logH,
                                    rfilter=rfilter,
                                    states=states,
                                    hopsize=steWin,
                                    winsize=lenWin)

# brutal normalization:
nsynth = synth / np.abs(synth).max()
wav.write('test.wav',sffhmm.samplingRate,
          np.int16(nsynth*(2**15)))
