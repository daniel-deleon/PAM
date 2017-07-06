import os
import numpy as np
import aifc
import wave
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib
import pandas as pd
import spectrogram_utilities
import soundfile as sf


""" plot spectrogram"""


def plot_spectrogram(audiopath, binsize=2**10, plotpath=None, colormap="jet"):

    info = sf.info(audiopath)
    sample_rate = info.samplerate
    duration = info.duration
    frames = info.frames

    with sf.SoundFile(audiopath, 'r') as f:
        _samples = f.read(frames)
        out_file = audiopath + '_spectrogram.jpg'
        plt.figure(figsize=(15, 7.5))
        spectrogram_utilities.plotstft(_samples, sample_rate, axis=None, binsize=2**10, plotpath=None)


# Set path to directory with folders train and test
path_data = '/Users/dannyd_sc/Google Drive/MBARI/PAM_Summer_Project_2017/BLED_Results/'

# Plot whale sound
audiopath = os.path.join(path_data, 'BlueWhaleB/BLED20150914/') + 'bb_sel.01.ch01.170620.083556.91..wav'
plot_spectrogram(audiopath, 0)

# Plot whale sound
#plot_spectrogram(path_data + 'bb_sel.109.ch01.170616.185530.36..wav', 1)

