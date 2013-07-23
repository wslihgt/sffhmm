"""sffhmm_hill_batch

This is the script is part of the files for the article:
    J.-L. Durrieu and J.-Ph. Thiran,
    'Source/Filter Factorial Hidden Markov Model,
    with Application to Pitch and Formant Tracking,'
    submitted, 2012.

Loads some pieces of information about the files from the
Hillenbrand dataset, available online at the address:
    http://homepages.wmich.edu/~hillenbr/voweldata.html

Provides:
    * loadHill: to load a file from hillenbrand's dataset
    * variables and dictionaries to make it easier to work
      with the provided .dat files
        timedatlabels : labels for columns of timedat (see below)
        bigdatlabels  : labels for columns of bigdat (-)
    * arrays that contain the content of these files
        timedat       : timing infos
        timedatnames  : filenames of the entries in timedat
        bigdat        : groundtruth for formants
        bigdatnames   : filenames for the entries of bigdat

Dependencies:
    * numpy, scipy (for wav file I/O)
    * you have to provide the correct path and files to the program
      by changing the script below (see comments for details)

"""

import numpy as np
#import scipy as sp
import scipy.io.wavfile as wav
import os

#############
# Warning ! #
#############
#
# local (and personal) path locations, please change the following 
# lines to point to the right files and locations, if you need to run
# this script:
if os.path.isdir('/Users/jeanlouis/work/BDD/'):
    timedataf = '/Users/jeanlouis/work/BDD/hillenbrand/vowels/timedata.dat'
    bigdataf = '/Users/jeanlouis/work/BDD/hillenbrand/vowels/bigdata.dat'
    prefixBDD = '/Users/jeanlouis/work/BDD/hillenbrand/vowels/'
elif os.path.isdir('/home/durrieu/work/BDD/'):
    timedataf = '/home/durrieu/work/BDD/hillenbrand/vowels/timedata.dat'
    bigdataf = '/home/durrieu/work/BDD/hillenbrand/vowels/bigdata.dat'
    prefixBDD = '/home/durrieu/work/BDD/hillenbrand/vowels/'
else:
    raise ImportError('Please provide the correct path and files, \n'+
                      'in loadHillenbrand.py.')

# loading the files and setting the corresponding labels:
timedat = np.loadtxt(timedataf, skiprows=6, usecols=(1,2,3,4))
timedatnames = np.loadtxt(timedataf, skiprows=6, usecols=(0,),
                          dtype=np.str)

timedatlabels = {'start':0, 'end':1,
                 'center1':2, 'center2':3}

bigdat = np.loadtxt(bigdataf, skiprows=43, usecols=range(1,30))
bigdatnames = np.loadtxt(bigdataf, skiprows=43, usecols=range(0,1),
                         dtype=np.str)

bigdatlabels = {'duration':0, 'F0':1, 'F1':2, 'F2':3, 'F3':4,
                'F1_10':5, 'F2_10':6, 'F3_10':7,
                'F1_20':8, 'F2_20':9, 'F3_20':10,
                'F1_30':11, 'F2_30':12, 'F3_30':13,
                'F1_40':14, 'F2_40':15, 'F3_40':16,
                'F1_50':17, 'F2_50':18, 'F3_50':19,
                'F1_60':20, 'F2_60':21, 'F3_60':22,
                'F1_70':23, 'F2_70':24, 'F3_70':25,
                'F1_80':26, 'F2_80':27, 'F3_80':28,
                }

def loadHill(filename, folder=prefixBDD):
    """fs, data = loadHill(filename, folder)
    
    INPUTS:
    -------
        filename is the name of the file to be processed
        folder is the path to the root folder for the database
            the program appends to `folder` the correct subfolder for the
            specified filename (which depends on the genre and age of
            the speaker)
    
    OUTPUTS: (same as wav.read)
    --------
        fs the sampling rate
        data the data array

    SEE ALSO:
    ---------
        wav.read, notably because the data type is usually int16, so you
        most likely need to convert to floats before processing the returned
        array.
        
    """
    if filename[0] == 'm':
        subpath = '/men/'
    elif filename[0] == 'w':
        subpath = '/women/'
    elif filename[0] == 'b' or filename[0] == 'g':
        subpath = '/kids/'
    filenameFull = str('').join([folder, subpath, filename, '.wav'])
    
    return wav.read(filenameFull)
