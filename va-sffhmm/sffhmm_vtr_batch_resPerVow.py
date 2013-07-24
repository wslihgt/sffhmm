"""Experiments for the VA-SFFHMM algorithm

Description
-----------
Runs the experiments for the VA-SFFHMM algorithm,
as described in:

 Durrieu, J.-L. and Thiran, J.-P.
 \"Source/Filter Factorial Hidden Markov Model,
 with Application to Pitch and Formant Tracking\"
 IEEE Transactions on Audio, Speech and Language Processing,
 Submitted Jan. 2013, Accepted July 2013.

Usage
-----
...

2013 - Jean-Louis Durrieu (http://www.durrieu.ch/research/)

"""
import sffhmm as fhmm
import scipy.io.wavfile as wav
import numpy as np
import warnings
import scikits.audiolab as al
import time

NaN = np.NaN

def mel2hz(f):
    return 700.0 * (10**(f / 2595.0) - 1)

def hz2mel(f):
    return 2595 * np.log10(1+f/700.0)

def twoAudio2htmlStr(audio1, audio2, width=100): 
    """
    outputs string to be written in an HTML file,
    corresponding to 1 line in a table, with 2
    columns: one per audio, with an HTML5 audio tag
    """
    htmlstring = '<tr>\n <td> '
    htmlstring += audio1.split('/')[-1]+'\n </td>\n' # assumes this is the seed
    htmlstring += ' <td> '
    htmlstring += '<audio width=%d src=%s' %(width, audio1)+\
                  ' controls=\"controls\">\n </td>\n' 
    htmlstring += ' <td> '
    htmlstring += '<audio width=%d src=%s' %(width, audio2)+\
                  ' controls=\"controls\">\n </td>\n</tr>' 
    return htmlstring

import matplotlib.pyplot as plt
from matplotlib.mlab import find as mlabfind

plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'nearest'

import sys
sys.path.append('../formants/')

import os
if os.path.isdir('/Users/jeanlouis/work/BDD/'):
    pathToArx = '/Users/jeanlouis/work/BDD/formants/'
    annotPath = '/Users/jeanlouis/work/BDD/formants/VTRFormants/'
    audioPath = '/users/jeanlouis/work/BDD/formants/timit/db/timit/'
                'programmation/python/hmm/'
elif os.path.isdir('/home/durrieu/work/BDD/'):
    pathToArx = '/home/durrieu/work/BDD/formants/'
    annotPath = '/home/durrieu/work/BDD/formants/VTRFormants/'
    audioPath = '/home/durrieu/work/BDD/formants/timit/db/timit/'

sys.path.append('../tools')
import speechTools as st
import manipTools as mt


files = mt.recursiveSearchFromRoot(annotPath, conditionExtension='.fb')

nbFilesHill = len(files)

numberGTFormants = 4

windowSizeInSeconds = 0.064
fs = 16000
lenWin = fhmm.nextpow2(np.ceil(fs * windowSizeInSeconds))
steWin = int(10. / 1000. *fs)# lenWin/4 # in accordance with annotation data
NFT = lenWin
maxFinFT = 8000
F = np.ceil(maxFinFT * NFT / np.double(fs))

comments = 'Changed the formant frequency transitions: '+\
           'x100 compared with learned parameters. keeping also '+\
           'all parameters for resynthesis, '+\
           'changing saved params to uint16. '+\
           'changed also some issues with odgd normalization'+\
           '\nAdded 20130517: stepSizeTransmat to adapt transition matrix'
chirpPerF0 = 1
ninner = 3
dispMat = True #False
debug = True #False
verbose = True
withNoiseFloor = True
withNoiseF0  = True
withFlatFilter = True
methPostInit = 'triangle' # 'lms'

formantsRange = {}
formantsRange[0] = [ 200.0, 1500.0] # check hillenbrand data
formantsRange[1] = [ 550.0, 3500.0]
formantsRange[2] = [1400.0, 4500.0]
formantsRange[3] = [2400.0, 6000.0] # adding one for full band
formantsRange[4] = [3300.0, 8000.0]
formantsRange[5] = [4500.0, 8000.0]
formantsRange[6] = [5500.0, 8000.0]

numberOfFormants = len(formantsRange)

sffhmm = fhmm.SFFHMM(samplingRate=fs,
                     NFT=NFT,
                     withNoiseF0=withNoiseF0,
                     withFlatFilter=withFlatFilter,
                     formantRanges=formantsRange,
                     F0range=[80,500],
                     chirpPerF0=chirpPerF0,
                     stepNotes=16,
                     withNoiseFloor=withNoiseFloor)

stepSizeTransmat =  steWin / np.double(fs)
##sffhmm.generateTransMatForSFModel(stepSize=steWin/np.double(fs))
# 20130517 DJL in order to get the transition harder, we can provide
# a step size which is smaller than it actually is
##stepSizeTransmat = 0.0001 * steWin / np.double(fs)
sffhmm.generateTransMatForSFModel(stepSize=stepSizeTransmat)

phonesAll = ['h#', 'q', 'ih', 'n', 'w', 'ey', 'dcl', 'jh', 'ix', 'gcl', \
             'g', 'ow', 'sh', 'iy', 's', 'epi', 'dh', 'd', 'tcl', 't', 'r', \
             'bcl', 'b', 'aa', 'z', 'eh', 'y', 'ux', 'nx', 'ng', 'el', 'hh', \
             'v', 'ao', 'pcl', 'p', 'pau', 'kcl', 'k', 'ah', 'm', 'l', 'axr', \
             'ae', 'dx', 'ay', 'f', 'ax', 'oy', 'uh', 'er', 'ax-h', 'ch', \
             'hv', 'th', 'en', 'aw', 'uw', 'eng', 'zh', 'em']

nbPhon = len(phonesAll)

# keeping in memory results
#   error matrices such that:
#     erfilmat[filenb][phoneme][targetFormant, EstimFormant] = (
#         average of error (MSE or absolute diff) for all frames of
#         current phoneme
#         )
erFilMatrix = {}
erAbsMatrix = {}
erAbsMatFrq = {}
voicing = {}
for n in range(nbFilesHill):
    erFilMatrix[n] = {}
    erAbsMatrix[n] = {}
    erAbsMatFrq[n] = {}
    voicing[n] = {}

# the computation times:
compTimes = np.zeros(nbFilesHill)
fileDuration = np.zeros(nbFilesHill)

import datetime
currentTime = datetime.datetime.strftime(\
        datetime.datetime.now(), format='%Y%m%dT%H%M')

savefilename = 'results/errorMatTimit_ninner-'+str(ninner)+\
               '_nbFP-'+str(numberOfFormants)+\
               '_chirpF0-'+str(chirpPerF0)+\
               '_lenWin-'+str(lenWin)+\
               '_steWin-'+str(steWin)+\
               '_noiseComp-'+str(withNoiseFloor)+\
               '_stepSizeTransmat-'+str(stepSizeTransmat)+\
               '_'+currentTime+\
               '.npz'

saveparamsfilename = 'results/paramsTimit_ninner-'+str(ninner)+\
                     '_nbFP-'+str(numberOfFormants)+\
                     '_chirpF0-'+str(chirpPerF0)+\
                     '_lenWin-'+str(lenWin)+\
                     '_steWin-'+str(steWin)+\
                     '_noiseComp-'+str(withNoiseFloor)+\
                     '_stepSizeTransmat-'+str(stepSizeTransmat)+\
                     '_'+currentTime+\
                     '.npz'

params = {'windowSizeInSeconds':windowSizeInSeconds,
          'fs':fs,
          'lenWin':lenWin,
          'steWin':steWin,
          'stepSizeTransmat':stepSizeTransmat,
          'NFT':NFT, 'maxFinFT':maxFinFT,
          'F':F, 'chirpPerF0':chirpPerF0,
          'ninner':ninner, 'withNoiseFloor': withNoiseFloor,
          'withNoiseF0': withNoiseF0,
          'withFlatFilter': withFlatFilter,
          'methPostInit': methPostInit,
          'currentTime':currentTime,
          'comments':comments}

htmlFilename = savefilename[:-3]+'html'
htmlFile = open(htmlFilename, 'w')
htmlFile.write(savefilename+"\n\n")
htmlFile.write("<table border=1>\n")


# From TIMIT doc phoncode.doc:
categories = ('vowels',
              'semivowels',
              'nasal',
              'fricatives',
              'affricatives',
              'stops')

cat2phon = {}
cat2phon['vowels'] = ('iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw',
                      'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw',
                      'ux', 'er', 'ax', 'ix', 'axr', 'ax-h')
cat2phon['semivowels'] = ('l','r','y','w','hh','hv','el')
cat2phon['nasal'] = ('m','n','ng','em','en','eng','nx')
cat2phon['fricatives'] = ('f', 'v','th','dh','s','z','sh','zh')
cat2phon['affricatives'] = ('ch','jh')
cat2phon['stops'] = ('b','d','g','p','t','k','d','x','q',
                     'pcl','bcl','tcl','dcl','kcl','gcl','dx')

silencePhones = ('pau', 'epi', 'h#',)
unvoicePhones = ('pau', 'epi', 'h#',) + \
                cat2phon['fricatives'] + \
                cat2phon['stops'] + \
                cat2phon['affricatives']

phon2cat = {}
for cat in categories:
    for ph in cat2phon[cat]:
        phon2cat[ph] = cat

for ph in phonesAll:
    if not(ph in phon2cat.keys()):
        print ph, 'is not taken into account in the error computations.'

#for filenb, vtrfile in enumerate([files[0]]):
for filenb, vtrfile in enumerate(files):
    print filenb, nbFilesHill, vtrfile
    suffix = vtrfile.split(annotPath)[-1].lower()[:-2]
    wavfile = audioPath+suffix+'wav'
    
    time0 = time.time()
    
    fs, data, metadata = st.readNISTwav(wavfile)
    data = data[0]
    if metadata['channel_count']!=1:
        warnings.warn("Channel number is "+\
                      "%d, we keep only first one."%metadata['channel_count'])
    data = np.double(data)
    fileDuration[filenb] = data.size / np.double(fs)
    data /= (abs(data)).max()
    obs, freqs, timeStamps = fhmm.stft(data, fs=fs, window=fhmm.hann(lenWin),
                                       hopsize=steWin, nfft=NFT)
    obs2 = np.log(np.array(abs(obs.T)**2, order='C'))
    eps = obs2[obs2>-np.Inf].min()#1e-20
    obs2 = np.maximum(obs2, eps)
    
    nframe, samPeriod, sampSize, numComps, fileType, trueFormants = \
        st.readFormantMS(vtrfile)
    trueFormants = trueFormants[:,:4].T
    
    mpost, states, posteriors = sffhmm.decode_var(obs=obs2,
                                                  n_innerLoop=ninner,
                                                  debug=debug,
                                                  thresholdEnergy=0,
                                                  postInitMeth=methPostInit,
                                                  verbose=verbose)
    logH = sffhmm._compute_energyComp(obs=obs2,
                                      posteriors=posteriors,
                                      debug=debug)
    
    rfilter = sffhmm._compute_recording_condition(obs=obs2,
                                                  posteriors=posteriors,
                                                  logH=logH,
                                                  debug=debug)
    
    mpost += logH + rfilter
    
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
        
    compTimes[filenb] = time.time() - time0
    # error measures per vowels
    #                          [numberGTFormants,
    #                           numberOfFormants,
    #                           nbFilesHill]
    txt = open(vtrfile[:-2]+'phn')
    phones = txt.readlines()
    txt.close()
    voicedFrames = np.array([True, ] * obs2.shape[0])
    for phnb, ph in enumerate(phones):
        ph_ = ph.strip('\n')
        elts = ph_.split(' ')
        # start and stop of phone, in frames
        start = int(1. * int(elts[0]) / steWin)#/np.double(fs)
        stop = int(1. * int(elts[1]) / steWin)#/np.double(fs)
        phon = elts[2]
        if phon in unvoicePhones:#silencePhones:
            voicedFrames[start:stop] = False
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
                np.sqrt(np.mean((statesInMel[1:][:,start:stop] - \
                    hz2mel(trueFormants[n, start:stop]*1000.))**2, axis=1))
            erAbsMatrix[filenb][str(phnb)+'_'+phon][n] = \
                np.mean(np.abs(statesInMel[1:][:,start:stop] - \
                    hz2mel(trueFormants[n, start:stop]*1000.)), axis=1)
            erAbsMatFrq[filenb][str(phnb)+'_'+phon][n] = \
                np.mean(np.abs(mel2hz(statesInMel[1:][:,start:stop]) - \
                    (trueFormants[n, start:stop]*1000.)), axis=1)
        print phon, erFilMatrix[filenb][str(phnb)+'_'+phon]\
            [range(numberGTFormants),\
             range(numberGTFormants)]
        
    # computing eval for voicing:
    voicing[filenb]['TP'] = (statesInNF[0][voicedFrames]>0).sum()
    voicing[filenb]['FP'] = (statesInNF[0][~voicedFrames]>0).sum()
    voicing[filenb]['TN'] = (statesInNF[0][~voicedFrames]==0).sum()
    voicing[filenb]['FN'] = (statesInNF[0][voicedFrames]==0).sum()
    print "    voicing", voicing[filenb]
    print "        accuracy", \
              (voicing[filenb]['TP']+voicing[filenb]['TN'])/\
              (1. * voicedFrames.size), \
              "and precision", \
              (voicing[filenb]['TP'])/\
              (1. * voicing[filenb]['TP'] + voicing[filenb]['FP'])
    print "    proc time:", compTimes[filenb],
    print "s, for file duration", fileDuration[filenb], "s"
    
    #print erFilMatrix[:,:,filenb][[range(4), range(4)]]
    #print erAbsMatrix[:,:,filenb][[range(4), range(4)]]
    #print erAbsMatFrq[:,:,filenb][[range(4), range(4)]]
    np.savez(savefilename,
             erAbsMatrix=erAbsMatrix,
             erFilMatrix=erFilMatrix,
             erAbsMatFrq=erAbsMatFrq,
             voicing=voicing,
             params=params)
        
    if dispMat:
        fig = plt.figure(10)
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax1.imshow(obs2.T)
        txt = open(wavfile[:-3]+'txt')
        readtext = txt.readline()
        txt.close()
        readtext = str(' ').join(readtext.strip().split(' ')[2:])
        ax1.set_title(str(filenb)+' '+wavfile.split('/')[-1] +\
                      ': observation\n' + readtext)
        for n in range(numberGTFormants): # Ground Truth
            ax1.plot(trueFormants[n]*1000./np.double(fs)*NFT,
                     'ok')
        for n in range(sffhmm.numberOfFormants+1):
            ax1.plot(statesInNF[n], '.')
        #.colorbar()
        ax1.plot(np.log(voicedFrames)+100, 'k', lw=2)
        ax1.plot(np.log([not(vv) for vv in voicedFrames])+100, 'w', lw=2)
        ax1.axis('tight')
        ax1.set_ylim([0,np.minimum(200,F)])
        ax2=fig.add_subplot(212, sharex=ax1,sharey=ax1)
        ax2.imshow(mpost.T)
        ax2.set_title('posterior mean: estimated spectrogram')
        ##for n in range(numberGTFormants): # Ground Truth
        ##    ax2.plot(trueFormants[n]*1000./np.double(fs)*NFT,
        ##             'ok')
        for n in range(sffhmm.numberOfFormants+1):
            ax2.plot(statesInNF[n], '.')
        #plt.colorbar()
        ax2.get_images()[0].set_clim(ax1.get_images()[0].get_clim())
        ax2.plot(np.log(voicedFrames)+100, 'k', lw=2)
        ax2.plot(np.log([not(vv) for vv in voicedFrames])+100, 'w', lw=2)
        ax2.axis('tight')
        ax2.set_ylim([0,np.minimum(200,F)])
        fig.canvas.draw()
        figname = audioPath+suffix+'_'+currentTime+'.pdf'
        plt.savefig(figname)
    
    oggfilename = wavfile[:-3]+'ogg'
    oggsyntname = wavfile[:-4]+'_vasffhmm_synth2.ogg'
    al.oggwrite(data=data,filename=oggfilename,fs=fs)
    #datas = sffhmm.soundSynthesis(obs=obs2,
    #                              states=states,
    #                              posteriors=posteriors,
    #                              hopsize=steWin,
    #                              winsize=lenWin)
    # Note: no need to resynthesize with the same rfilter!
    #     one can indeed set rfilter to 0, and the result should
    #     still sound the same
    datas = sffhmm.soundSynthesisParams(logH=logH,
                                        rfilter=rfilter,
                                        states=states,
                                        hopsize=steWin,
                                        winsize=lenWin,)
    datasynthParams = wavfile[:-4]+'_'+currentTime+'.npz'
    np.savez(datasynthParams, logH=logH, rfilter=rfilter,
             states=[np.uint16(states[n]) \
                     for n in range(len(states))],
             steWin=steWin, lenWin=lenWin)
    datas = datas / (abs(datas).max()+.01)
    al.oggwrite(data=datas,filename=oggsyntname,fs=fs)
    
    htmlFile.write(twoAudio2htmlStr(oggfilename, oggsyntname))
    
    htmlFile.flush()

htmlFile.write("</table>\n")
htmlFile.close()

# loading the data:
##savefilename = 'errorMatTimit_ninner-3_nbFP-7_chirpF0-1_lenWin-1024_'+\
##               'steWin-160_20120228T1503.npz'
# 10 inner loops, update of estimate noise with fixed step
##savefilename = 'errorMatTimit_ninner-10_nbFP-7_chirpF0-1_lenWin-1024_'+\
##               'steWin-160_noiseComp-True_20120414T2055.npz' 
# 10 inner loops, update estimate noise, almost no changes to noise estimate 
## savefilename = 'errorMatTimit_ninner-10_nbFP-7_chirpF0-1_lenWin-1024_'+\
##                'steWin-160_noiseComp-True_20120414T2033.npz'
# 10 inner loops, fixed noise
#savefilename = 'errorMatTimit_ninner-10_nbFP-7_chirpF0-1_lenWin-1024_'+\
#               'steWin-160_noiseComp-True_20120423T1430.npz'
#struc = np.load(savefilename)
#erFilMatrix = struc['erFilMatrix'].tolist()
#erAbsMatFrq = struc['erAbsMatFrq'].tolist()
#struc.close()

# COMPUTING RESULTS AND FIGURES # 

fontsize = 12

nbins = 100

figwidth, figheight = [9,5]

errorPerPh = {}
for ph in phonesAll:
    errorPerPh[ph] = [0, np.zeros([numberGTFormants,
                                   numberOfFormants])]

errorPerCat = {}
for cat in categories:
    errorPerCat[cat] = [0, np.zeros([numberGTFormants,
                                     numberOfFormants])]

erAbsPerPh = {}
for ph in phonesAll:
    erAbsPerPh[ph] = [0, np.zeros([numberGTFormants,
                                   numberOfFormants])]

erAbsPerCat = {}
for cat in categories:
    erAbsPerCat[cat] = [0, np.zeros([numberGTFormants,
                                     numberOfFormants])]

for filenb, vtrfile in enumerate(files):
    # print filenb, nbFilesHill
    vtrfile = files[filenb]
    suffix = vtrfile.split(annotPath)[-1].lower()[:-2]
    wavfile = audioPath+suffix+'wav'
    phones = ['dummy',]*len(erFilMatrix[filenb].keys())
    for key in erFilMatrix[filenb].keys():
        phnb, ph = key.split('_')
        phones[int(phnb)] = ph
    for phnb, ph in enumerate(phones):
        phon = ph
        if not(np.any(np.isnan(erFilMatrix[filenb][str(phnb)+'_'+phon]))):
            errorPerPh[phon][0] += 1
            errorPerPh[phon][1] += erFilMatrix[filenb][str(phnb)+'_'+phon]
            erAbsPerPh[phon][0] += 1
            erAbsPerPh[phon][1] += erAbsMatFrq[filenb][str(phnb)+'_'+phon]
            if phon in phon2cat.keys():
                cat = phon2cat[phon]
                errorPerCat[cat][0] += 1
                errorPerCat[cat][1] += erFilMatrix[filenb][str(phnb)+'_'+phon]
                erAbsPerCat[cat][0] += 1
                erAbsPerCat[cat][1] += erAbsMatFrq[filenb][str(phnb)+'_'+phon]

for ph in phonesAll:
    errorPerPh[ph][1] /= errorPerPh[ph][0]
    erAbsPerPh[ph][1] /= erAbsPerPh[ph][0]
for cat in categories:
    errorPerCat[cat][1] /= errorPerCat[cat][0]
    erAbsPerCat[cat][1] /= erAbsPerCat[cat][0]

plt.figure(1)
plt.clf()
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
#for phnb, ph in enumerate(phonesAll):
phnb = 0
phonCatOrder = []
for cat in categories:
    for ph in cat2phon[cat]:
        if ph in phonesAll:
            if not(np.any(np.isnan(errorPerPh[ph][1][range(4),range(4)]))):
                for n in range(4):
                    plt.plot(phnb, errorPerPh[ph][1][n,n],
                             marker=marker[n],
                             color=color[n])
            phnb+=1
            phonCatOrder.append(ph)

plt.xticks(np.arange(phnb),phonCatOrder,rotation=90)
plt.legend(('F1', 'F2','F3','F4'))
plt.xlim([-1, phnb])
plt.title("Average Mean Squared Error for each phoneme, VA-SFFHMM")
plt.draw()

plt.figure(2)
plt.clf()
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
for phnb, cat in enumerate(categories):
    if not(np.any(np.isnan(errorPerCat[cat][1][range(4),range(4)]))):
        for n in range(4):
            plt.plot(phnb, errorPerCat[cat][1][n,n],
                     marker=marker[n],
                     color=color[n])
plt.xticks(np.arange(len(categories)),categories,rotation=45)
plt.legend(('F1', 'F2','F3','F4'))
plt.xlim([-1, len(categories)])
plt.title("Mean Squared Error for each phoneme category, VA-SFFHMM")
plt.draw()

plt.figure(3)
plt.clf()
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
#for phnb, ph in enumerate(phonesAll):
phnb = 0
phonCatOrder = []
for cat in categories:
    for ph in cat2phon[cat]:
        if ph in phonesAll:
            if not(np.any(np.isnan(erAbsPerPh[ph][1][range(4),range(4)]))):
                for n in range(4):
                    plt.plot(phnb, erAbsPerPh[ph][1][n,n],
                             marker=marker[n],
                             color=color[n])
            phnb+=1
            phonCatOrder.append(ph)

plt.xticks(np.arange(phnb),phonCatOrder,rotation=90)
plt.legend(('F1', 'F2','F3','F4'))
plt.xlim([-1, phnb])
plt.title("Mean Absolute Difference for each phoneme, VA-SFFHMM")
plt.draw()

top = 0.95
bottom = 0.18
left = 0.05
right = 0.98
rotation = 70
ha = 'right' # horizontal alignment of xticks
plt.rcParams['lines.markersize'] = 12

fig = plt.figure(4, figsize=(12,6))
plt.clf()
plt.subplot(144)
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
for phnb, cat in enumerate(categories):
    if not(np.any(np.isnan(erAbsPerCat[cat][1][range(4),range(4)]))):
        for n in range(4):
            plt.plot(phnb, erAbsPerCat[cat][1][n,n],
                     marker=marker[n],
                     color=color[n])
plt.xticks(np.arange(len(categories)),categories,rotation=rotation,ha=ha)
plt.yticks(visible=False)
plt.grid()
plt.legend(('F1', 'F2','F3','F4'))
if withNoiseFloor:
    plt.title('VA-SFFHMM, noise compensation')
else:
    plt.title('VA-SFFHMM')
plt.xlim([-1, len(categories)])
plt.draw()

erAbsPerCatMSR = {}
erAbsPerCatMSR['vowels'] = (64,105,125)
erAbsPerCatMSR['semivowels'] = (83,122,154)
erAbsPerCatMSR['nasal'] = (67,120,112)
erAbsPerCatMSR['fricatives'] = (129,108,131)
erAbsPerCatMSR['affricatives'] = (141,129,149)
erAbsPerCatMSR['stops'] = (130,113,119)
erAbsPerCatMSR['liquids'] = (NaN, NaN, NaN)

erAbsPerCatUsr = {}
erAbsPerCatUsr['vowels'] = (55,69,84)
erAbsPerCatUsr['semivowels'] = (68,80,103)
erAbsPerCatUsr['nasal'] = (75,112,106)
erAbsPerCatUsr['fricatives'] = (91,113,125)
erAbsPerCatUsr['affricatives'] = (89,118,135)
erAbsPerCatUsr['stops'] = (91,110,116)
erAbsPerCatUsr['liquids'] = (NaN, NaN, NaN)

erAbsPerCatWav = {}
erAbsPerCatWav['vowels'] = (70,94,154)
erAbsPerCatWav['semivowels'] = (89,126,222)
erAbsPerCatWav['nasal'] = (96,229,239)
erAbsPerCatWav['fricatives'] = (209,263,439)
erAbsPerCatWav['affricatives'] = (292,407,390)
erAbsPerCatWav['stops'] = (168,210,286)
erAbsPerCatWav['liquids'] = (NaN, NaN, NaN)

ax1 = fig.get_axes()[0]
ax2 = fig.add_subplot(141, sharey=ax1)
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
for phnb, cat in enumerate(categories):
    if not(np.any(np.isnan(erAbsPerCatUsr[cat]))):
        for n in range(3):
            plt.plot(phnb, erAbsPerCatUsr[cat][n],
                     marker=marker[n],
                     color=color[n])
plt.xticks(np.arange(len(categories)),categories,rotation=rotation,ha=ha)
#plt.yticks(visible=False)
plt.grid()
plt.title('Labeler Variations')
plt.ylabel('Mean Absolute Diff. per frame (Hz)')
plt.legend(('F1', 'F2','F3'))
plt.xlim([-1, len(categories)])
ax2 = fig.add_subplot(142, sharey=ax1)
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
for phnb, cat in enumerate(categories):
    if not(np.any(np.isnan(erAbsPerCatWav[cat]))):
        for n in range(3):
            plt.plot(phnb, erAbsPerCatWav[cat][n],
                     marker=marker[n],
                     color=color[n])
plt.xticks(np.arange(len(categories)),categories,rotation=rotation,ha=ha)
plt.yticks(visible=False)
plt.grid()
plt.title('Wavesurfer')
plt.legend(('F1', 'F2','F3'))
plt.xlim([-1, len(categories)])
ax2 = fig.add_subplot(143, sharey=ax1)
marker = ['o', 'D', 's', '.']
color = ['b', 'g', 'r', 'k']
for phnb, cat in enumerate(categories):
    if not(np.any(np.isnan(erAbsPerCatMSR[cat]))):
        for n in range(3):
            plt.plot(phnb, erAbsPerCatMSR[cat][n],
                     marker=marker[n],
                     color=color[n])
plt.xticks(np.arange(len(categories)),categories,rotation=rotation,ha=ha)
plt.title('MSR [Deng et al., 2006]')
plt.legend(('F1', 'F2','F3'))
plt.xlim([-1, len(categories)])
plt.subplots_adjust(top=top,bottom=bottom,
                    left=left,right=right)
plt.yticks(visible=False)
plt.grid()
plt.ylim([0, 600])
plt.draw()
plt.savefig(savefilename[:-3]+'pdf')

### images for TASLP 2013 article

plt.rc('lines', linewidth=4)
Fs = sffhmm.samplingRate * 1.
NFT = sffhmm.NFT * 1.
FHzmax = 4000.0  
nFmax = np.int32(np.ceil(FHzmax/Fs*NFT))
ytickslab = np.array([1000, 2000, 3000, 4000])
ytickspos = np.int32(np.ceil(ytickslab/Fs*NFT))
fontsize = 16
figheight=4.5
figwidth=9.0

nbF0 = 200
plt.figure(figsize=[figwidth, figheight])
plt.plot((sffhmm.means[0][nbF0,:nFmax, ]), color='k')
plt.xticks(ytickspos, ytickslab, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.axis('tight')
plt.xlabel('Frequency (Hz)', fontsize=fontsize)
plt.legend((['F0 = %dHz' %(sffhmm.F0Table[nbF0])]))
plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
# plt.savefig('w0.pdf')
# not quite the image from the article, see SFSNMF code instead

plt.figure(figsize=[figwidth, figheight])
lsRange = ('-', '--', ':')
p = 3 # formant number
step = 2
nbElts = 4
legendThing = {}
for nn in range(nbElts):
    plt.plot(sffhmm.means[p][nn*step, :nFmax] -
             np.max(sffhmm.means[p][nn*step, :nFmax]) ,
             lsRange[(nn)%len(lsRange)],
             color='k')
    plt.xticks(ytickspos, ytickslab, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    legendThing[nn] = u'%1.2f' %(sffhmm.poleAmp[p*sffhmm.nElPerFor+nn*step])

plt.legend(legendThing.values(), loc='best')
plt.xlabel('Frequency (Hz)', fontsize=fontsize)
plt.axis('tight')
plt.subplots_adjust(top=.96, right=0.96, left=.06, bottom=.13)
# plt.savefig('wp_sffhmm.pdf')
