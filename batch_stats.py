#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Combine all predicted results and plot
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import os
import pandas as pd
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

if __name__ == '__main__':

    years = range(2015, 2017)
    months = range(8, 12)
    dates = []
    prefix = 'BlueWhaleD'
    base_directory = '/Volumes/PAM_Analysis/BatchDetections/BLED'
    for y in years:
      for m in months:
        # directory where predicted output should be stored
        predict_dir = '{0}/{1}/{2}/{3:02}'.format(base_directory, prefix, y, m)
        all_files = glob.iglob(predict_dir + "**/predictions.csv", recursive=True)
        df_new = pd.concat((pd.read_csv(f,index_col=None, header=0) for f in all_files))
        for index, row in df_new.iterrows():
            base = os.path.basename(row.Filename)
            date = datetime.strptime(base.split('.')[0], '%Y%m%dT%H%M%S')
            sample_start_msec = 1e3 * int(base.split('.')[1])  / 250
            date += timedelta(milliseconds=sample_start_msec)
            dates.append(date)

        df = df.append(df_new, ignore_index=True)

    # # reindex with dates
    df.index = dates

    # pivot on predictions to simplify plotting
    df_sorted = df.sort_index()
    pivoted = df_sorted.pivot(index=None, columns='Predicted')

    # replace score with total, rename the score column to total
    pivoted = pivoted[pivoted.Score >= 0.8]
    pivoted = pivoted.rename(columns={'Score': 'Total'})

    pivoted.to_csv('{0}_all_predictions.csv'.format(prefix))
    pivoted.describe().to_csv('{0}_all_statistics.csv'.format(prefix));

    width = 10
    height = 6

    # bar plot the weekly summary
    df_pivot = pivoted.resample('7D').sum()
    index = df_pivot.index.map(lambda t: t.strftime('%Y-%m-%d'))
    df_pivot.index = index
    df_pivot.plot(kind='bar', alpha=0.75, rot=45, figsize=(width, height), width=1.5)
    plt.xlabel('Week')
    plt.ylabel('Total Calls')
    plt.title('Weekly Calls for ' + ','.join(prefix))
    plt.tight_layout()
    # plt.show()
    plt.savefig('{0}/{1}.png'.format(os.getcwd(), '{0}_weekly_predictions_7D_resample_sum'.format(prefix)))
    plt.close()

    df_pivot.to_csv('{0}_weekly_predictions_7D_resample_sum.csv'.format(prefix))

    exit(-1)


    months = [8, 9, 10, 11, 12]
    prefix = ['BlueWhaleB', 'FinWhale']

    base_directory = '/Volumes/PAM_Analysis/BatchDetections/BLED'
    df = pd.DataFrame()
    dates = []
    for p in prefix:
        # directory where predicted output should be stored
        predict_dir = '{0}/{1}/'.format(base_directory, p)
        all_files = glob.iglob(predict_dir + "**/predictions.csv", recursive=True)
        df_new = pd.concat((pd.read_csv(f,index_col=None, header=0) for f in all_files))
        for index, row in df_new.iterrows():
            base = os.path.basename(row.Filename)
            date = datetime.strptime(base.split('.')[0], '%Y%m%dT%H%M%S')
            sample_start_msec = 1e3 * int(base.split('.')[1])  / 250
            date += timedelta(milliseconds=sample_start_msec)
            dates.append(date)

        df = df.append(df_new, ignore_index=True)

    # # reindex with dates
    df.index = dates

    # pivot on predictions to simplify plotting
    df_sorted = df.sort_index()
    pivoted = df_sorted.pivot(index=None, columns='Predicted')

    # replace score with total, rename the score column to total
    pivoted = pivoted[pivoted.Score >= 0.9]
    pivoted = pivoted.rename(columns={'Score': 'Total'})

    pivoted.to_csv('all_predictions.csv')
    pivoted.describe().to_csv('all_statistics.csv');
    
    width = 10
    height = 6

    # bar plot the weekly summary
    df_pivot = pivoted.resample('7D').sum()
    index = df_pivot.index.map(lambda t: t.strftime('%Y-%m-%d'))
    df_pivot.index = index
    df_pivot.plot(kind='bar', alpha=0.75, rot=45, figsize=(width, height), width=1.5)
    plt.xlabel('Week')
    plt.ylabel('Total Calls')
    plt.title('Weekly Calls for ' + ','.join(prefix))
    plt.tight_layout()
    #plt.show()
    plt.savefig('{0}/{1}.png'.format(os.getcwd(), 'weekly_predictions_7D_resample_sum'))
    plt.close()

    df_pivot.to_csv('weekly_predictions_7D_resample_sum.csv')

