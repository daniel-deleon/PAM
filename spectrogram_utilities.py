#!/usr/bin/env python
# coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)


    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1

    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
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


# Function to allow code reusability
def plot_spectrogram(ax, P):
    from matplotlib import mlab, cm
    import matplotlib.ticker as plticker

    plt.imshow(P, origin='lower', extent=[-6, 6, -1, 1], aspect=4, cmap=cm.get_cmap('bwr'))
    loc = plticker.MultipleLocator(base=3.0)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set_xticklabels(np.arange(-0.5, 2.5, 0.5))
    ax.set_yticklabels(range(0, 1001, 250))
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=12)


""" plot spectrogram"""


def plotstft(samples, sample_rate, axis, binsize=2 ** 10, plotpath=None, colormap="jet"):
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    fig = plt.figure()
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))

    if axis is None:
        plt.colorbar()
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins - 1])
        plt.ylim([0, freqbins])
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / sample_rate])
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    else:
        axis.set_xlabel("time (s)")
        axis.set_ylabel("frequency (hz)")
        axis.set_xlim([0, timebins - 1])
        axis.set_ylim([0, freqbins])
        axis.set_xticks(xlocs,
                        ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / sample_rate])
        axis.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs]),

    if plotpath is None:
        plt.show()
    else:
        extent = plt.axes().get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig(plotpath, bbox_inches=extent)
        fig.savefig(bbox_inches='tight', pad_inches=0.0)
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
    plot_spectrogram(ax1, R)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, P)
    plt.suptitle('Comparison low frequency range (left) vs full image (right)', fontsize=16)
    plt.show()

    # Extreme values are capped to mean ± 1.5 std
    fact_ = 1.50
    Q[Q > mval + fact_ * sval] = mval + fact_ * sval
    Q[Q < mval - fact_ * sval] = mval - fact_ * sval

    # Let's see it
    R = Q.copy()
    R[:minM] = 34
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(121)
    plot_spectrogram(ax1, P)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, R)
    plt.suptitle('Comparison original image (left) vs capped extreme values (right)', fontsize=16)
    plt.show()

    print('Done')