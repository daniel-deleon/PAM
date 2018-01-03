from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import spectrogram_utilities
import soundfile as sf
import os
import glob
import pandas as pd
import conf
import cv2

def preprocess_raw(wav_path, conf):
    '''
    Converts wav files in a directory to spectrograms. Be default creates a directory called spectrograms in the 
    same parent directory as the wav files
    :param wav_path: path to the wav files
    :param conf: spectrogram configuration settings
    :return: 
    '''
    for wav in sorted(glob.iglob(wav_path + '**/*.wav', recursive=True)):

        # get the selection number in the filename
        wav_path, wav_file = os.path.split(wav)
        spec_path = '{0}/spectrogram_raw'.format(wav_path.split('wav')[0])
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

def preprocess_training(base_path, conf):
    '''
    Generate spectrograms of all sounds recursively in a directory
    First search for the txt that describes the annotation - use this to sort the spectrograms into
    separate directories for training the classifier

    :param base_path:  base directory where all produced spectrograms, wav files and detection files are stored
    :param conf:  configuration settings for generating the spectrogram
    '''  
    
    for full_filename in sorted(glob.iglob(base_path + '**/*selections.txt', recursive=True)):
        print('Reading {0}'.format(full_filename))
        df = pd.read_csv(full_filename, sep='\t')
        if 'Selection' in df.keys():
            path, file = os.path.split(full_filename)
            date_str = file.split('_')[0]
            date_start = datetime.strptime(date_str, '%Y%m%dT%H%M%SZ')
            #for wav in glob.iglob('{0}/wav/*_{1:%Y%m%dT%H%M%S}*.wav'.format(base_path, date_start), recursive=False):
            for wav in sorted(glob.iglob('{0}/wav/{1:%Y%m%dT%H%M%S}*.wav'.format(path, date_start), recursive=True)):

                # get the selection number in the filename
                wav_path, wav_file = os.path.split(wav)
                selection = int(wav_file.split('.')[4])

                # lookup class by selection number
                match = df[df.Selection == selection]

                if 'classification' in df.keys():
                    label = match.iloc[0].classification
                if 'Label' in df.keys():
                    label = match.iloc[0].Label

                if pd.isnull(label):
                  continue

                if 'bdt' in label:
                    continue
                if 'wff' in label:
                    continue

                spectrogram_path_by_class = os.path.join(base_path, 'spectrogram', label)

                if not os.path.exists(spectrogram_path_by_class):
                    os.makedirs(spectrogram_path_by_class)
 
                info = sf.info(wav)
                sample_rate = info.samplerate
                frames = info.frames

                with sf.SoundFile(wav, 'r') as f:
                    _samples = f.read(frames) 
                    _, file = os.path.split(wav)
                    base_file = file.split('.wav')[0]
                    out_file = os.path.join(spectrogram_path_by_class, base_file + '.spectrogram.png')
                    out_file_jpg = os.path.join(spectrogram_path_by_class, base_file + '.spectrogram.jpg')
                    if not os.path.exists(out_file_jpg):
                        spectrogram_utilities.optimize_spectrogram_blob(conf, _samples, sample_rate, plotpath=out_file)

                    # uncomment below to display instead of just saving to disk
                    #spectrogram_utilities.display_optimized_spectrogram(conf, _samples, sample_rate, binsize=2 ** 10, plotpath=out_file)
                 

if __name__ == '__main__':

    # TODO: refactor this code into command arguments
    #blued_path = '/Volumes/PAM_Analysis/TrainingData/BlueWhaleD//'
    #preprocess_training(blued_path, conf.BLUE_D)

    #blued_path = '/Volumes/PAM_Analysis/TestData/BlueWhaleD/20160804T070000Z/'
    #preprocess_raw(blued_path, conf.BLUE_D)


    #fin_path = '/Volumes/PAM_Analysis/TrainingData/FinWhale20Hz/'
    #preprocess_training(fin_path, conf.FIN_20HZ)

    #blue_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/BlueWhaleD/2016/'
    #fin_bled_path = '/Volumes/PAM_Analysis/Batch_Detections/BLED/FinWhale/2015/'
    #for m in months:
    #    preprocess_raw('{0}/{1:02}/wav/'.format(fin_bled_path, m), conf.FIN)

    # Set path to directory with folders train and test wav files 
    blue_bled_path = '/Volumes/PAM_Analysis/BatchDetections/BLED/BlueWhaleD/'
    #fin_bled_path = '/Volumes/PAM_Analysis/BatchDetections/BLED/FinWhale/2016/'

    years = range(2015, 2017)
    months = range(1, 12)
    for y in years:
      blue_bled_path = '/Volumes/PAM_Analysis/BatchDetections/BLED/BlueWhaleD/{0}'.format(y)
      for m in months:
          preprocess_raw('{0}/{1:02}/wav/'.format(blue_bled_path, m), conf.BLUE_D)
