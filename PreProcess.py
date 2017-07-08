import matplotlib.pyplot as plt
import spectrogram_utilities
import soundfile as sf
import os
  

def plot_spectrogram(audio_path, spectrogram_out=None, bin_size=2 ** 10):
    '''
     plot spectrogram"
    :param audio_path: 
    :param spectrogram_out: 
    :param bin_size: 
    :param colormap: 
    :return: 
    '''

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

        #spectrogram_utilities.optimize_spectrogram(_samples, sample_rate, binsize=bin_size, plotpath=out_file)
        # uncomment below to display instead of just saving to disk
        spectrogram_utilities.display_optimized_spectrogram(_samples, sample_rate, binsize=bin_size, plotpath=out_file)


if __name__ == '__main__':

    # Set path to directory with folders train and test wav files
    path_data = '/Users/dannyd_sc/Google Drive/MBARI/PAM_Summer_Project_2017/BLED_Results/' 
    
    # Set path to directory to save optimized spectrogram of wav files
    spectrogram_path = os.path.join(os.getcwd(), 'spectrogram')
    
    # make the directory if it doesn't exist
    if not os.path.exists(spectrogram_path):
        os.mkdir(spectrogram_path)
    
    # Plot whale sound
    audio_path = os.path.join(path_data, 'BlueWhaleB/BLED20150914/') + 'bb_sel.01.ch01.170620.083556.91..wav' 
    plot_spectrogram(audio_path, spectrogram_path, bin_size=250)
    
    # Generate spectrograms of all sounds recursively in a directory  
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                plot_spectrogram(audio_path, spectrogram_path, 250)
    
     
