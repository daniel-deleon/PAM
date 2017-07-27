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
from pandas import ExcelWriter
from pandas import ExcelFile

table = pd.read_excel('/Users/ddeleon/Desktop/misclassifiedFixed.xlsx', sheetname='Sheet1')

for i in table.index:
    if table['Actual'][i] == table['Predicted'][i]:
        print(i)