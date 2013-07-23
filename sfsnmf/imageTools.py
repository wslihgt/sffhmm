#!/usr/bin/python
#
# a script to define some matlab compatible image functions

# copyright 2010 Jean-Louis Durrieu
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

import matplotlib.pyplot as plt
import numpy as np

# The following instructions define some characteristics for the figures
# In order to be able to use latex formulas in legends and text in
# figures:
# plt.rc('text', usetex=True)
# Turn on interactive mode to display the figures:
plt.ion()
# Characteristics of the figures:
fontsize = 20
linewidth = 4
markersize = 16
# Setting the above characteristics as defaults:
plt.rc('legend',fontsize=fontsize)
plt.rc('lines',markersize=markersize)
plt.rc('lines',lw=linewidth)

def imageM(*args,**kwargs):
    """
    imageM(*args, **kwargs)

    This function essentially is a wrapper for the
    matplotlib.pyplot function imshow, such that the actual result
    looks like the default that can be obtained with the MATLAB
    function image.

    The arguments are the same as the arguments for function imshow.
    """
    # The appearance of the image: nearest means that the image
    # is not smoothed:
    kwargs['interpolation'] = 'nearest'
    # keyword 'aspect' allows to adapt the aspect ratio to the
    # size of the window, and not the opposite (which is the default
    # behaviour):
    kwargs['aspect'] = 'auto'
    kwargs['origin'] = 0
    plt.imshow(*args,**kwargs)

def subplotMatrix(matrixToPlot, figureHandler=None,
                  titleMatrix=None, mappingX=None,
                  xlimits=None, ylimits=None, **kwargs):
    """
    subplotMatrix(matrixToPlot, figureHandler=None,
                  titleMatrix=None, mappingX=None,
                  xlimits=None, ylimits=None, **kwargs):

    plots the matrix matrixToPlot, with vectors matrixToPlot[:,n]
    in different subplots.
    
    """
    if figureHandler == None:
        h=plt.figure()
    else:
        h=plt.figure(figureHandler.number)
        plt.hold(True)
        
    F, N = matrixToPlot.shape
    nl = 1
    nc = N
    while(nc > 1.5 * nl):
        nl = nl + 1.0
        nc = N / nl
    nc = np.int32(np.ceil(nc))
    nl = np.int32(nl)
    
    if nc > 8 or nl > 5:
        print "Not drawing, too many to draw: nc=", nc, "nl=", nl
    
    for n in range(N):
        plt.subplot(nl, nc, n+1)
        plt.plot(matrixToPlot[:,n], **kwargs)
        plt.axis('tight')
        if mappingX != None:
            xtickspos, xtickslab = plt.xticks()
            plt.xticks(xtickspos[1:-1], np.int32(mappingX[np.int32(xtickspos[1:-1])]))

        if xlimits != None:
            plt.xlim(xlimits)
        if ylimits != None:
            plt.ylim(ylimits)
        if titleMatrix!=None:
            plt.title(titleMatrix[n])
        
    return h

def plotRegions(startTimes, endTimes, vmin=0, vmax=1, labels=None):
    nbRegions = startTimes.size
    for n in range(nbRegions):
        line = plt.plot([startTimes[n], startTimes[n]],
                        [vmin, vmax], '-')
        plt.plot([endTimes[n], endTimes[n]],
                 [vmin, vmax], '-', color=plt.getp(line[0], 'color'))
        ## plt.plot([startTimes[n], startTimes[n]],
        ##          [vmin, vmax], 'g-')
        ## plt.plot([endTimes[n], endTimes[n]],
        ##          [vmin, vmax], 'y-')
        if labels != None:
            plt.text((startTimes[n]+endTimes[n])/2.0,
                     vmax - (- vmin + vmax) * 1.0 / 10.0 - \
                     (-vmin+vmax) * 8.0 / 10.0 / (nbRegions-1.0) * n,
                     labels[n],
                     color='w',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=16,
                     weight='extra bold',
                     backgroundcolor=plt.getp(line[0], 'color'))

def plotRegionsX(startTimes, endTimes, vmin=0, vmax=1, labels=None):
    nbRegions = startTimes.size
    for n in range(nbRegions):
        plt.plot([startTimes[n], startTimes[n]],
                 [vmin, vmax], 'g-')
        plt.plot([endTimes[n], endTimes[n]],
                 [vmin, vmax], 'y-')
    if labels != None:
        for n in range(nbRegions):
            plt.text((startTimes[n]+endTimes[n])/2.0,
                     vmax - (- vmin + vmax) * 1.0 / 10.0 - \
                     (-vmin+vmax) * 8.0 / 10.0 / (nbRegions-1.0) * n,
                     labels[n],
                     color='k',
                     horizontalalignment='center',
                     verticalalignment='baseline',
                     fontsize=16,
                     weight='extra bold')

def plotRegionsY(startTimes, endTimes, vmin=0, vmax=1, backgroundcolor='pink', color='k', labels=None):
    nbRegions = startTimes.size
    for n in range(nbRegions):
        plt.plot([vmin, vmax],
                 [startTimes[n], startTimes[n]],
                 'g-')
        plt.plot([vmin, vmax],
                 [endTimes[n], endTimes[n]],
                 'y-')
    if labels != None:
        for n in range(nbRegions):
            plt.text(vmax - (- vmin + vmax) * 1.0 / 10.0 - \
                     (-vmin+vmax) * 8.0 / 10.0 / (nbRegions-1.0) * n,
                     (startTimes[n]+endTimes[n])/2.0,
                     labels[n],
                     color=color,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=16,
                     weight='extra bold',
                     backgroundcolor=backgroundcolor)

def plotRectangle(bottomLeftCorner, topRightCorner, color='b', **kwargs):
    x0, y0 = bottomLeftCorner
    x1, y1 = topRightCorner
    plt.plot([x0, x0], [y0, y1], color=color, **kwargs)
    plt.plot([x0, x1], [y0, y0], color=color, **kwargs)
    plt.plot([x0, x1], [y1, y1], color=color, **kwargs)
    plt.plot([x1, x1], [y0, y1], color=color, **kwargs)
    

def plotNotes(notes, onsets, durations, width=None, color='b', **kwargs):
    nbNotes = len(notes)
    if width==None:
        width = 0.5 * np.ones([nbNotes])
    plt.hold(True)
    for n in range(nbNotes):
        bottomLeftCorner = np.array([onsets[n], np.double(notes[n]) - width[n]])
        topRightCorner = np.array([np.double(onsets[n])+np.double(durations[n]), np.double(notes[n]) + width[n]])
        plotRectangle(bottomLeftCorner, topRightCorner, color=color, **kwargs)

def plotVar(tableArrays, fignum=None, **kwargs):
    """plots the data like in R plot(data)"""
    
    nVar = len(tableArrays)
    plt.figure(fignum)
    for i in range(nVar):
        for j in range(i+1, nVar):
            plt.subplot(nVar-1, nVar-1, i*(nVar-1)+j)
            plt.plot(tableArrays[j], tableArrays[i], '.', **kwargs)

def plotVarMat(tableArrays, **kwargs):
    """plots the data like in R plot(data)"""
    
    nVar = tableArrays.shape[1]
    plt.figure()
    for i in range(nVar):
        for j in range(i+1, nVar):
            plt.subplot(nVar-1, nVar-1, i*(nVar-1)+j)
            plt.plot(tableArrays[:,j], tableArrays[:,i], '.', **kwargs)

def imagesSharedAxes(images, orientation='vertical', fig=None):
    nbIm = len(images)
    if fig is None:
        fig = plt.figure()
    
    fig.clear()
    ax = []
    if orientation=='vertical':
        ax.append(fig.add_subplot(nbIm, 1, 1))
        for n in range(1, nbIm):
            ax.append(fig.add_subplot(nbIm, 1, n+1,
                                      sharex=ax[0], sharey=ax[0]))
    elif orientation=='horizontal':
        ax.append(fig.add_subplot(1, nbIm, 1))
        for n in range(1, nbIm):
            ax.append(fig.add_subplot(1, nbIm, n+1,
                                      sharex=ax[0], sharey=ax[0]))
    else:
        raise ValueError("no such orientation:" + str(orientation))
    
    for n in range(nbIm):
        ax[n].imshow(images[n])
    
    plt.draw()
    return fig, ax, 
