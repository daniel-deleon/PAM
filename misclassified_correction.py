#!/usr/bin/env python

__author__ = "Daniel De Leon"
__copyright__ = "Copyright 2017, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Daniel De Leon"
__email__ = "ddeleon at mbari.org"
__doc__ = '''

Places misclassifcations in correct directory using misclassifcation spreadsheet

'''

import pandas as pd
import shutil
import os

table = pd.read_excel('/Volumes/PAM_Analysis/Classifiers/InceptionV3/models/BlueWhaleBUpdate/01_80test/misclassifiedFixed.xlsx', sheetname='Sheet1')

data_type = 'BlueWhaleBUpdate'
base_dir = '/Volumes/PAM_Analysis/TrainingData/data/' + data_type

for i in table.index:
    if table['Actual'][i] == table['Predicted'][i]:
        misclassified = table['Filename'][i]
        prediction = table['Predicted'][i]

        filename = misclassified.split('/')[-1]
        dst_base = misclassified.split(data_type)[-1]

        source = '{0}{1}'.format(base_dir, dst_base)
        dst = '{0}/{1}/{2}'.format(base_dir, prediction, filename)

        if os.path.exists(source):
            shutil.move(source, dst)