import matplotlib.pyplot as plt
import spectrogram_utilities
import soundfile as sf
import os
import glob
import pandas as pd
import conf
 

def preprocess_raw(wav_path, conf):
    '''
    Converts wav files in a directory to spectrograms. Be default creates a directory called spectrograms in the 
    same parent directory as the wav files
    :param wav_path: path to the wav files
    :param conf: spectrogram configuration settings
    :return: 
    '''
    for wav in glob.iglob(wav_path + '**/*.wav', recursive=True):

        # get the selection number in the filename
        wav_path, wav_file = os.path.split(wav)
        spec_path = '{0}/spectrogram'.format(wav_path.split('wav')[0])
        if not os.path.exists(spec_path):
            os.mkdir(spec_path)

        info = sf.info(wav)
        sample_rate = info.samplerate
        frames = info.frames

        with sf.SoundFile(wav, 'r') as f:
            _samples = f.read(frames)
            _, file = os.path.split(wav)
            base_file = file.split('.wav')[0]
            out_file = os.path.join(spec_path, base_file + '.spectrogram.png')
            if not os.path.exists(out_file):
                spectrogram_utilities.optimize_spectrogram(conf, _samples, sample_rate, plotpath=out_file)

def preprocess(bled_path, spectrogram_path, conf):
    '''
    Generate spectrograms of all sounds recursively in a directory
    First search for the txt that describes the annotation - use this to sort the spectrograms into
    separate directories for training the classifier

    :param bled_path:  BLED detection path
    :param spectrogram_path:  directory where all produced spectrograms are stored
    :param conf:  configuration settings for generating the spectrogram
    '''  
    
    for full_filename in glob.iglob(bled_path + '**/*.txt', recursive=True):
        print('Reading {0}'.format(full_filename))
        df = pd.read_csv(full_filename, sep='\t')
        if 'Selection' in df.keys():
            path, file = os.path.split(full_filename)
            date = path.rsplit('/BLED', 1)[-1]
            for wav in glob.iglob(path + '**/*.wav', recursive=False):

                # get the selection number in the filename
                wav_path, wav_file = os.path.split(wav)
                selection = int(wav_file.split('.')[1])

                # lookup class by selection number
                match = df[df.Selection == selection]
                label = match.iloc[0].Label
                spectrogram_path_by_class = os.path.join(spectrogram_path, label)

                if not os.path.exists(spectrogram_path_by_class):
                    os.mkdir(spectrogram_path_by_class)
 
                info = sf.info(wav)
                sample_rate = info.samplerate
                frames = info.frames

                with sf.SoundFile(wav, 'r') as f:
                    _samples = f.read(frames) 
                    _, file = os.path.split(wav)
                    base_file = file.split('.wav')[0]
                    out_file = os.path.join(spectrogram_path_by_class, base_file + date + '.spectrogram.png')

                    spectrogram_utilities.optimize_spectrogram(conf, _samples, sample_rate, plotpath=out_file)
                    # uncomment below to display instead of just saving to disk
                    # spectrogram_utilities.display_optimized_spectrogram(_samples, sample_rate, binsize=bin_size, plotpath=out_file)
                 

if __name__ == '__main__':

    # TODO: refactor this code into command arguments

    blue_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/BlueWhaleB/2015/'
    fin_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/FinWhale/2015/'

    months = [10]
    for m in months:
        preprocess_raw('{0}/{1:02}/wav/'.format(blue_bled_path, m), conf.BLUE_B)
    for m in months:
        preprocess_raw('{0}/{1:02}/wav/'.format(fin_bled_path, m), conf.FIN)

    # Set path to directory with folders train and test wav files 
    blue_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/BlueWhaleB/2016/'
    fin_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/FinWhale/2016/'

    months = [8, 9, 11]
    for m in months:
        preprocess_raw('{0}/{1:02}/wav/'.format(blue_bled_path, m), conf.BLUE_B)
    for m in months:
        preprocess_raw('{0}/{1:02}/wav/'.format(fin_bled_path, m), conf.FIN)
