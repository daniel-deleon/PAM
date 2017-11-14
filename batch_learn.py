#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Runs transfer learning training on spectrogram images from the MBARI hydrophone
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import subprocess

if __name__ == '__main__':
    learning_rate=.01
    options = '--learning_rate {0} --random_crop 20 --testing_percentage 20'.format(learning_rate)
    prefix = ['BlueWhaleD'] #'FinWhale', 'BlueWhaleB']
    base_directory = '/Volumes/PAM_Analysis/TrainingData'
    exemplar_dir = '/Volumes/PAM_Analysis/TrainingData/exemplars/'
    model_base_dir = '/Volumes/PAM_Analysis/Classifiers/InceptionV3/models/'

    for p in prefix:
      # image directory where cropped images are located
      image_dir = '{0}/{1}/spectrogram'.format(base_directory, p)

      # This is the directory the bottleneck features are generated; bottleneck features are generated by running each image through
      # the inception model. Once these are generated, they are cached.
      bottleneck_dir = '{0}/{1}/bottlenecks'.format(base_directory, p)

      # output directory to store model
      model_dir = '{0}/{1}/{2}'.format(model_base_dir, p, str(learning_rate).split('.')[1])

      # add in any additional options here
      options += ' --image_dir {0} --bottleneck_dir {1} --model_dir {2} --exemplar_dir {3}'.format(image_dir,
                                                                                              bottleneck_dir,
                                                                                              model_dir,
                                                                                              exemplar_dir)

      cmd = 'python ./learn.py {0}'.format(options)
      print(cmd)
      subproc = subprocess.Popen(cmd, env=os.environ, shell=True)
      subproc.communicate()

    print('Done')
    exit(-1)