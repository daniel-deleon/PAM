#!/usr/bin/env python

__author__ = "Danelle Cline, Daniel De Leon"
__copyright__ = "Copyright 2017, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Daniel De Leon"
__email__ = "ddeleon at mbari.org"
__doc__ = '''

Utility module for generating spectrograms for classification tests

@author: __author__
@status: __status__
@license: __license__
'''

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib import mlab

from scipy.ndimage.filters import gaussian_filter

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    '''
    short time fourier transform of audio signal
    :param sig: 
    :param frameSize: 
    :param overlapFac: 
    :param window: 
    :return: 
    '''
    P = matplotlib.mlab.specgram(sig,
                             NFFT=frameSize,
                             Fs=250,
                             window=window,
                             noverlap=int(overlapFac*250),
                             pad_to = None,
                             sides = 'default',
                                scale_by_freq = None)
    return P

def logscale_spec(spec, sr=44100, factor=20.):
    '''
     scale frequency axis logarithmically 
    :param spec: 
    :param sr: 
    :param factor: 
    :return: 
    '''
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)


    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def plot_spectrogram(ax, P, colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len):
    '''
    Function to allow code reuse
    :param ax: 
    :param P: 
    :return: 
    '''
    from matplotlib import mlab
    import matplotlib.ticker as plticker

    plt.imshow(P, origin='lower', cmap=colormap)
    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))

    ax.set_xticks(xlocs)
    ax.set_yticks(ylocs)

    ax.set_xlim([0, timebins - 1])
    ax.set_ylim([0, freqbins])
    ax.set_xticklabels(["%.02f" % l for l in ((xlocs * sample_len / timebins) + (0.5 * binsize)) / sample_rate])
    ax.set_yticklabels(["%.02f" % freq[i] for i in ylocs])

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=12)


def optimize_spectrogram(samples, sample_rate, binsize=2 ** 10, colormap=cm.get_cmap('bwr'), plotpath=os.path.join(os.getcwd())):
    '''
    optimize and save spectrogram
    :param samples: 
    :param sample_rate: 
    :param binsize: 
    :param plotpath: 
    :param colormap: 
    :return: 
    '''
    from matplotlib import mlab, cm
    s = stft(samples, binsize, 0.80)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    P = np.transpose(ims)
    freq_bin = float(P.shape[0]) / float(sample_rate / 2)  # bin/Hz
    cut_off_freq = 29
    minM = -1 * (P.shape[0] - int(cut_off_freq * freq_bin))
    Q = P.copy()
    mval, sval = np.mean(Q[:minM]), np.std(Q[:minM])

    # Extreme values are capped to mean ± 1.5 std
    fact_ = 1.50
    Q[Q > mval + fact_ * sval] = mval + fact_ * sval
    Q[Q < mval - fact_ * sval] = mval - fact_ * sval

    # Save the final result, slicing only the part of the array above the cut-off frequency cut_off_freq and blurring
    # make a 3x3 figure without the frame
    fig = plt.figure()
    width = 3
    height = 3
    fig.set_size_inches(width, height)
    plt.axis('off')
    blurred = gaussian_filter(Q, sigma=1)
    plt.imshow(blurred[minM:, :], origin="lower", aspect=2, cmap=colormap)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
    extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plotpath, bbox_inches=extent)
    fig.clf()
    print('Done creating ' + plotpath)

def display_optimized_spectrogram(samples, sample_rate, binsize=2 ** 10, plotpath=None):
    '''
    optimize spectrogram ans and optionally save spectrogram
    :param samples: 
    :param sample_rate:  
    :param binsize: 
    :param plotpath: 
    :param colormap: 
    :return: 
    '''
    from matplotlib import mlab, cm
    s = stft(samples, binsize, 0.80)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    colormap = cm.get_cmap('bwr')
    sample_len = len(samples)

    fig = plt.figure()
    ax0 = plt.subplot(121)
    plot_spectrogram(ax0, np.transpose(ims), colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    plt.show()
    fig.clf()

    P = np.transpose(ims)
    freq_bin = float(P.shape[0]) / float(sample_rate / 2)  # bin/Hz
    cut_off_freq = 29  # Hz
    minM = -1 * (P.shape[0] - int(cut_off_freq * freq_bin))
    Q = P.copy()
    R = Q.copy()
    mval, sval = np.mean(Q[:minM]), np.std(Q[:minM])
    R[:minM] = 34  # 68/2 maxfreq/2
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(121)
    plot_spectrogram(ax1, R, colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, P, colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    plt.suptitle('Comparison low frequency range (left) vs full image (right)', fontsize=16)
    plt.show()
    fig.clf()

    # Extreme values are capped to mean ± 1.5 std
    fact_ = 1.50
    Q[Q > mval + fact_ * sval] = mval + fact_ * sval
    Q[Q < mval - fact_ * sval] = mval - fact_ * sval

    # Let's see it
    R = Q.copy()
    R[:minM] = 34
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(121)
    plot_spectrogram(ax1, P, colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, R, colormap, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    plt.suptitle('Comparison original image (left) vs capped extreme values (right)', fontsize=16)
    plt.show()
    fig.clf()

    # Save the final result, slicing only the part of the array above the cut-off frequency cut_off_freq and blurring
    if plotpath is not None:
        # make a 3x3 figure without the frame
        fig = plt.figure()
        width = 3
        height = 3
        fig.set_size_inches(width, height)
        plt.axis('off')
        blurred = gaussian_filter(R[minM:, :], sigma=1)
        plt.imshow(blurred, origin="lower", aspect=4, cmap=colormap)
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
        extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(plotpath, bbox_inches=extent)
        fig.clf()
        print('Done creating ' + plotpath)
