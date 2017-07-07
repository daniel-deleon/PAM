import matplotlib.pyplot as plt
import spectrogram_utilities
import soundfile as sf
import os

""" plot spectrogram"""


def plot_spectrogram(audio_path, spectrogram_out=None, bin_size=2**10, colormap="jet"):

    info = sf.info(audio_path)
    sample_rate = info.samplerate
    frames = info.frames

    with sf.SoundFile(audio_path, 'r') as f:
        _samples = f.read(frames)
        out_file = ''
        if spectrogram_out:
            _, file = os.path.split(audio_path)
            base_file = file.split('.wav')[0]
            out_file = os.path.join(spectrogram_out, base_file + 'spectrogram.png')

        spectrogram_utilities.optimize_spectrogram(_samples, sample_rate, binsize=2**10, plotpath=out_file)

# Set path to directory with folders train and test wav files
path_data = '/Users/dannyd_sc/Google Drive/MBARI/PAM_Summer_Project_2017/BLED_Results/'

# Set path to directory to save optimized spectrogram of wav files
spectrogram_path = os.path.join(os.getcwd(), 'spectrogram')
#spectrogram_out = os.path.join(path_data, 'spectrogram')
# make the directory if it doesn't exist
if not os.path.exists(spectrogram_path):
    os.mkdir(spectrogram_path)

# Plot whale sound
audio_path = os.path.join(path_data, 'BlueWhaleB/BLED20150914/') + 'bb_sel.01.ch01.170620.083556.91..wav'
#audio_path = '/Users/dcline/Downloads/bb_sel.542.ch01.170617.024759.58..wav'
plot_spectrogram(audio_path, spectrogram_path)

# Plot whale sound
#plot_spectrogram(path_data + 'bb_sel.109.ch01.170616.185530.36..wav', 1)

