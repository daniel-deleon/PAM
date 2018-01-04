import matplotlib.pyplot as plt
import spectrogram_utilities
import soundfile as sf
import os
import glob
import pandas as pd
import conf
from datetime import datetime, timedelta
from time import strftime


def generate_wav(bled_path, wav_dir, pad_seconds):
    '''
    Generate wav clips of all sounds recursively in a directory
    Search for the txt that describes the annotation and generate wav clips padded

    :param bled_path:  BLED detection path 
    :param wav_dir:  base directory for all wav files 
    :param pad_seconds:  padding in seconds to add to beginning and ending of wav files 
    '''

    for full_filename in glob.iglob(bled_path + '**/*Table01.txt', recursive=True):
        print('Reading {0}'.format(full_filename))
        df = pd.read_csv(full_filename, sep='\t')
        if 'Selection' in df.keys() \
            and 'Begin Time (s)' in df.keys() \
            and 'End Time (s)' in df.keys():

            path, file = os.path.split(full_filename)
            wav_path = os.path.join(path, 'wav')
            if not os.path.exists(wav_path):
                os.mkdir(wav_path)

            a = file.split('.')[0]
            dts = a.split('_')
            date_start = datetime.strptime(dts[0], '%Y%m%dT%H%M%SZ')

            # get the decimated file for this period
            wav_file = '{0}/{1:04}/{2:02}/{3}.d1024.250Hz.wav'.format(wav_dir, date_start.year, date_start.month, '_'.join(dts))

            start_time_secs = df['Begin Time (s)']
            end_time_secs = df['End Time (s)']
            selections = df['Selection']

            if not os.path.exists(wav_file):
                print('Decimated wav file {0} does not exist'.format(wav_file))
                continue

            with sf.SoundFile(wav_file, 'r') as f:
                info = sf.info(wav_file)
                pad_samples = int(pad_seconds * info.samplerate)
                max_samples = int(info.duration * info.samplerate)

                for i, row in df.iterrows():
                    start_sample = int(start_time_secs[i] * info.samplerate)
                    end_sample = int(end_time_secs[i] * info.samplerate)
                    selection = selections[i]

                    start_datetime = date_start + timedelta(milliseconds=int(start_time_secs[i]*1000))

                    if start_sample >= pad_samples:
                        start_sample -= pad_samples
                        start_datetime -= timedelta(pad_seconds)
                    if end_sample + pad_samples <= max_samples:
                        end_sample += pad_samples

                    start_iso = date_start.strftime("%Y%m%dT%H%M%S")
                    new_file = '{0}/{1}.{2}.{3}.sel.{4:02}.ch01.wav'.format(wav_path, start_iso, start_sample, end_sample, selection)

                    if not os.path.exists(new_file):
                        try:
                            f.seek(start_sample)
                            duration_sample = end_sample - start_sample
                            samples = f.read(duration_sample)
                            print('Creating {0}'.format(new_file))

                            with sf.SoundFile(file=new_file, mode='w',samplerate=info.samplerate, channels=info.channels,
                                              subtype=info.subtype) as fout:
                                fout.info = info
                                fout.write(samples)
                        except Exception as ex:
                            print(ex)
                            print('{0} {1} {2} {3}'.format(wav_file, start_sample, end_sample, max_samples))
                        continue

    print('Done generating wav')
if __name__ == '__main__':
    wav_dir = '/Volumes/PAM_Analysis/decimated/'
    years = range(2015, 2018)
    months = range(12, 13)
    for y in years:
      blue_bled_path = '/Volumes/PAM_Analysis/BatchDetections/BLED/BlueWhaleD/{0:4}/'.format(y)
      for m in months:
          generate_wav('{0}/{1:02}'.format(blue_bled_path, m), wav_dir, conf.BLUE_D['padding_secs'])

    print('Done')