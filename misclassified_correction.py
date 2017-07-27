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

table = pd.read_excel('/Users/ddeleon/Desktop/misclassifiedFixed.xlsx', sheetname='Sheet1')


for i in table.index:
    if table['Actual'][i] == table['Predicted'][i]:
        misclass = table['Filename'][i]
        shutil.move(misclass, misclass.replace(misclass.split('/')[-2],table['Actual'][i]))