#!/usr/bin/env python

__author__ = "Danelle Cline, Daniel De Leon"
__copyright__ = "Copyright 2017, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
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
import subprocess
import conf

from matplotlib import cm
from matplotlib import mlab

from scipy.ndimage.filters import gaussian_filter
import cv2
from viscm import viscm

COLORMAP=conf.parula_map

def stft(sig, NFFT, overlapFac=0.5):
    '''
    short time fourier transform of audio signal
    :param sig: array of amplitude values from audio file
    :param NFFT: number of ffts
    :param overlapFac: percentage of window overlap
    :return: transpose of values that make up spectrogram
    '''
    P, freqs, bins = matplotlib.mlab.specgram(sig,
                                NFFT=NFFT,
                                Fs=250,
                                detrend=mlab.detrend_none,
                                window=mlab.window_hanning,
                                noverlap=int(overlapFac*NFFT))
    return np.transpose(P)

def logscale_spec(spec, sr=44100, factor=20.):
    '''
     scale frequency axis logarithmically 
    :param spec: initial values that make up spectrogram
    :param sr: samples/sec
    :param factor: exponential growth rate factor
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


def plot_spectrogram(ax, P, timebins, freqbins, freq, binsize, sample_rate, sample_len):
    '''
    Function to allow code reuse
    :param ax: 
    :param P: 
    :return: 
    '''
    from matplotlib import mlab
    import matplotlib.ticker as plticker

    plt.imshow(P, origin='lower', cmap=COLORMAP)
    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    ax.set_xticks(xlocs)
    ax.set_yticks(ylocs)

    ax.set_xlim([0, timebins - 1])
    ax.set_ylim([39, 51])
    ax.set_xticklabels(["%.02f" % l for l in ((xlocs * sample_len / timebins) + (0.5 * binsize)) / sample_rate])
    ax.set_yticklabels(["%.02f" % freq[i] for i in ylocs])

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    #cbar = plt.colorbar()
    #cbar.set_label('Amplitude', fontsize=12)
    plt.axis('off')


def optimize_spectrogram_blob(conf, samples, sample_rate, plotpath=os.path.join(os.getcwd()), imshow = False):
    '''
    optimize and save spectrogram
    :param conf:  configuration settings for the spectrogram
    :param samples:  samples to create spectrogram from
    :param plotpath: path to the file to save the output to
    :return:
    '''
    s = stft(samples, conf['num_fft'], .95)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = np.where(sshow>0., 20.*np.log10(np.abs(sshow))/ 10e-6, 0.)

    P = np.transpose(ims)
    freq_bin = float(P.shape[0]) / float(sample_rate / 2)
    minM = -1 * (P.shape[0] - int(conf['low_cut_off_freq'] * freq_bin))
    maxM = -1 * (P.shape[0] - int(conf['high_cut_off_freq'] * freq_bin))

    found = False
    for factor in np.linspace(1, 3, num=9):
        Q = P.copy()
        Q[:minM] = Q[maxM:] = 34
        mval, sval = np.mean(Q), np.std(Q)
        if found:
            print('Found object')
            break;
        smooth_normalize(Q, conf, factor, maxM, minM, mval, plotpath, sval)
        image = cv2.imread(plotpath)
        image2 = cv2.imread(plotpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # find contours; should only be on object since the subplots create a frame around the images
        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        tlx, tly, w, h = cv2.boundingRect(cnt)
        # crop the image and run object detection
        crop_img = image[tly:tly + h, tlx: tlx + w]
        ret2, th = cv2.threshold(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        # if found an object, center the image on this
        if imshow:
            cv2.imshow('otsu', th)
            cv2.waitKey(500)
        for j in range(0, 3):
            th = cv2.erode(cv2.dilate(th, kernel3), kernel2)
            if imshow:
                cv2.imshow('cleaned', th)
                cv2.waitKey(500)
            # only adjust the time (x) dimension - keep the frequency dimension the same
            found, tlx, _, w, _ = find_object(th, image2)
            if found:
                final_crop_img = image[tly:tly + h, tlx: tlx + w]
                #cv2.imshow('final', final_crop_img)
                #cv2.waitKey(500)
                cv2.imwrite(plotpath, final_crop_img)
                break

    path, file = os.path.split(plotpath)
    filename = file.split('.png')[0]
    plotpath_jpeg = '{0}/{1}.jpg'.format(path, filename)

    # and convert to jpeg and resize
    cmd = "/usr/local/bin/convert '{0}' -adaptive-resize 299x299\! '{1}'".format(plotpath, plotpath_jpeg)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
    subproc.communicate()
    print('Done creating ' + plotpath_jpeg)
    os.remove(plotpath)


def smooth_normalize(Q, conf, factor, maxM, minM, mval, plotpath, sval):
    '''
    Smooth and normalize the spectrogram array. Creates an image and writes to a file
    :param Q: the spectrogram array
    :param conf:  configuration settings
    :param factor:  factor to multiply to standard deviation for normalization
    :param maxM: max index in the array to keep
    :param minM: min index in the array to keep
    :param mval: mean value for normalization
    :param plotpath: path to file to save the results to (png file)
    :param sval: standard deviation for normalization
    :return:
    '''
    # Extreme values are capped to mean ± factor* std
    Q[Q > mval + factor * sval] = mval + factor * sval
    Q[Q < mval - factor * sval] = mval - factor * sval
    # uncomment below to zoom in whale call
    Q = Q[minM:maxM]
    # pad with zeros along the edges to deal with boundary effects from the convolution
    zero_pad = 1
    npad = ((0, 0), (zero_pad, zero_pad))
    Q = np.pad(Q, pad_width=npad, mode='constant', constant_values=0)
    # Smooth and remove edges
    inner = 3
    filter = np.ones(inner)
    if conf['blur_axis'] is 'time':
        Q2 = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=Q)
    elif conf['blur_axis'] is 'frequency':
        Q2 = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=0, arr=Q)
    else:
        Q2 = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=Q)
    num_cols = Q2.shape[1] - zero_pad - 1
    num_rows = Q2.shape[0] - zero_pad - 1
    Q = Q2[zero_pad + 1:num_rows, zero_pad + 1:num_cols]
    # Save the final result, slicing only the part of the array above the cut-off frequency cut_off_freq and blurring
    # make a 3x3 figure without the frame
    fig = plt.figure()
    width = 3
    height = 3
    fig.set_size_inches(width, height)
    plt.axis('off')
    plt.imshow(np.flipud(Q), interpolation='bilinear', cmap=COLORMAP)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)
    extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # plt.show()
    plt.savefig(plotpath, bbox_inches=extent)
    plt.close()


def optimize_spectrogram(conf, samples, sample_rate, plotpath=os.path.join(os.getcwd())):
    '''
    optimize and save spectrogram
    :param conf:  configuration settings for the spectrogram
    :param samples:  samples to create spectrogram from
    :param plotpath: path to the file to save the output to
    :return: 
    '''
    s = stft(samples, conf['num_fft'], .95)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = np.where(sshow>0., 20.*np.log10(np.abs(sshow))/ 10e-6, 0.)

    P = np.transpose(ims)
    freq_bin = float(P.shape[0]) / float(sample_rate / 2)
    minM = -1 * (P.shape[0] - int(conf['low_cut_off_freq'] * freq_bin))
    maxM = -1 * (P.shape[0] - int(conf['high_cut_off_freq'] * freq_bin))

    Q = P.copy()
    Q[:minM] = Q[maxM:] = 34

    # Extreme values are capped to mean ± factor* std
    mval, sval = np.mean(Q), np.std(Q)

    smooth_normalize(Q, conf, 1, maxM, minM, mval, plotpath, sval)

    path, file = os.path.split(plotpath)
    filename = file.split('.png')[0]
    plotpath_jpeg = '{0}/{1}.jpg'.format(path, filename)

    # and convert to jpeg and resize
    cmd = "/usr/local/bin/convert '{0}' -adaptive-resize 299x299\! '{1}'".format(plotpath, plotpath_jpeg)
    subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
    subproc.communicate()
    print('Done creating ' + plotpath_jpeg)
    os.remove(plotpath)


def display_optimized_spectrogram(conf, samples, sample_rate, binsize=2 ** 10, plotpath=None):
    '''
    optimize spectrogram ans and optionally save spectrogram
    :param samples: array of dB values from audio file
    :param sample_rate:  samples/sec of audio
    :param binsize: bins/sec of audio
    :param plotpath: location of produced spectrogram
    :return: enhanced spectrogram
    '''
    from matplotlib import mlab, cm
    s = stft(samples, binsize, 0.95)
    sshow, freq = logscale_spec(s, factor=1.0, sr=sample_rate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    sample_len = len(samples)

    fig = plt.figure()
    ax0 = plt.subplot(121)
    plot_spectrogram(ax0, np.transpose(ims), timebins, freqbins, freq, binsize, sample_rate, sample_len)
    fig.clf()

    P = np.transpose(ims)
    freq_bin = float(P.shape[0]) / float(sample_rate / 2) # Hz/bin
    minM = -1 * (P.shape[0] - int(conf['low_cut_off_freq'] * freq_bin))
    maxM = -1 * (P.shape[0] - int(conf['high_cut_off_freq'] * freq_bin))
    Q = P.copy()
    R = Q.copy()
    mval, sval = np.mean(np.append(Q[:minM], [Q[maxM:]])), np.std(np.append(Q[:minM], [Q[maxM:]]))
    R[:minM] = R[maxM: - 1] = 34  # 68/2 maxfreq/2
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(121)
    plot_spectrogram(ax1, R, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, P, timebins, freqbins, freq, binsize, sample_rate, sample_len)
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
    plot_spectrogram(ax1, P, timebins, freqbins, freq, binsize, sample_rate, sample_len)
    ax2 = plt.subplot(122)
    plot_spectrogram(ax2, R, timebins, freqbins, freq, binsize, sample_rate, sample_len)
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
        plt.imshow(blurred, origin="lower", aspect=4, cmap=COLORMAP)
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1.1, top=1.0, bottom=0)
        extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(plotpath, bbox_inches=extent)
        fig.clf()
        print('Done creating ' + plotpath)


def find_object(image_bin, image_color, imshow = False):
  # get blobs
  im, contours, heirachy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  img = cv2.drawContours(image_color, contours, -1, (0, 255, 0), 7)

  # if only one contour, no objects found. Should have at least background and one object
  if len(contours) == 1:
    return False, -1, -1, -1, -1

  if imshow:
    cv2.namedWindow('object', cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    cv2.imshow('object', image_bin)
    cv2.waitKey()
    cv2.imshow('object', img)
    cv2.waitKey()

  # keep valid contours
  valid_contours = []
  cnt = 0
  for c in contours:
    try:
      # get rectangle bounding contour
      [x, y, w, h] = cv2.boundingRect(c)
      area = cv2.contourArea(c)
      #print("area {0} x {1} y {2} w {3} h {4}".format(area, x, y, w, h))
      # get valid areas, not blobs along the edge or noise
      if area > 2000 and area < 15000:
        pt = [float(y), float(y)]
        cn = contours[cnt]
        valid_contours.append(c)
        if imshow:
          img = cv2.drawContours(image_color, [cn], 0, (0, 0, 255), 5)
          cv2.imshow('object', img)
          cv2.waitKey()
      cnt += 1

    except Exception as ex:
      print(ex)

  if imshow:
    cv2.waitKey()
    for i in range(0,2):
      cv2.destroyAllWindows()

  if len(valid_contours) > 0:
    # find largest contour in mask
    c = max(valid_contours, key=cv2.contourArea)
    [x, y, w, h] = cv2.boundingRect(c)
    return True, x, y, w, h

  return False, -1, -1, -1, -1
