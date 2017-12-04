#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Utility class for miscellaneous learning plot functions

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

# comment this if you want to display plots, e.g. use the plt.show() function
import matplotlib
matplotlib.use('Agg')

import os
import math
import numpy as np
import fnmatch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.gridspec as gridspec
import pandas as pd

from matplotlib.lines import Line2D

linestyles = ['-', '--', ':']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

def find_matches(model_out_dir, filename):
  matches = []
  for root, dirnames, filenames in os.walk(model_out_dir):
    for filename in fnmatch.filter(filenames, filename):
      matches.append(os.path.join(root, filename))

  return matches

def plot_metrics(model_out_dir):

  try:

    # find all matches
    csv_file = os.path.join(model_out_dir, '_acc_prec_all.csv')
    matches = find_matches(model_out_dir, 'metrics.csv')

    # combine all the high-level data into a csv file
    header_out = False
    with open(csv_file, 'w') as fout:
      for metrics_file in sorted(matches):
        if os.path.exists(metrics_file):
          with open(metrics_file) as fin:
            header = next(fin)
            if not header_out:
                fout.write(header)
                header_out = True
            for line in fin:
                fout.write(line)

    # plot in a bar chart
    df = pd.read_csv(csv_file, sep=',')
    ax = df.plot(kind='bar', title="Metrics\n" + model_out_dir, figsize=(12,10))
    ax.set_xticklabels(df.Distortion, rotation=90)
    ax.set_ylim([0.0, 1.05])
    for p in ax.patches:
      ax.annotate(str(np.round(p.get_height(), decimals=4)), (p.get_x() * 1.005, p.get_height() * 1.005),fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(model_out_dir, os.path.join(model_out_dir,  '_acc_prec_all.png')), format='png', dpi=120);
    plt.close('all')

    # combine all the class-level data into a csv file
    matches = find_matches(model_out_dir,  'metrics_by_class.csv')
    csv_file = os.path.join(model_out_dir,  'acc_prec_by_class.csv')
    header_out = False
    with open(csv_file, 'w') as fout:
      for metrics_file in sorted(matches):
        if os.path.exists(metrics_file):
          with open(metrics_file) as fin:
            header = next(fin)
            if not header_out:
                fout.write(header)
                header_out = True
            for line in fin:
                fout.write(line)

    # plot each distortion accuracy/prediction plots in a combo bar/line plot
    df_metrics = pd.read_csv(csv_file, sep=',')
    distortions = list(set(df_metrics.Distortion))
    for v in sorted(distortions):
      df = df_metrics[df_metrics['Distortion'] == v]
      if not df.empty:
        df = df.sort_values(by='Accuracy', ascending=False)
        ax = df[['Accuracy','Precision']].plot(kind='bar', title="Metrics by Class " + v + "\n" + model_out_dir, figsize=(12,10))
        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), df[['NumTrainingImages']].values, linestyle='-', marker='o', linewidth=2.0)
        ax.set(xlabel='Class')
        ax.set_ylim([0.0, 1.05])
        ax2.set(ylabel='Total Training Example Images')
        ax2.set_yscale('log')
        ax.set_xticklabels(df.Class, rotation=90)
        # don't plot the bar values if too many categories - this is cluttered
        if len(ax.patches) < 20:
          for p in ax.patches:
            ax.annotate(str(np.round(p.get_height(),decimals=2)), (p.get_x() * 1.005, p.get_height() * 1.005),fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir,  '_acc_prec_by_class_' + v + '.png'), format='png', dpi=120);
        plt.close('all')

    # plot each distortion ROC curve
    matches = find_matches(model_out_dir,  'metrics_roc.pkl')
    max_per_plot = 15
    for cm_file in matches:
      df = pd.read_pickle(cm_file)

      # Compute ROC curve and ROC area for each class
      if not df.empty:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()
        y_test = np.vstack(df['y_test'].values)
        y_score = np.vstack(df['y_score'].values)
        labels = df['labels'].values[0] # only need the first row here
        num_classes = y_test.shape[1]
        distortion = cm_file.split('/')[-2]
        if num_classes >= max_per_plot:
          num_plots = int(math.ceil(float(num_classes) / float(max_per_plot)))
        else:
          num_plots = 1

        # stack the grids vertically if more than one plot
        if num_plots > 2:
          fig = plt.figure(figsize=(22, 16));
        else:
          fig = plt.figure(figsize=(12, 10));

        a = 0; j = 0;
        k = int(math.ceil(float(num_plots) / float(2)))
        if num_plots > 5:
          gs = gridspec.GridSpec(2,k)
        else:
          gs = gridspec.GridSpec(num_plots,1)

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in range(num_classes):
          if (i + max_per_plot) % max_per_plot == 0:
            ax = fig.add_subplot(gs[a])
            ax.set_xlim([0, 30])
            ax.set_ylim([80, 100.1])
            if a == 0:
              ax.set_xlabel('False Positive Rate (FPR)')
              ax.set_ylabel('True Positive Rate (TPR)')
              ax.set_title('Receiver Operating Characteristic (ROC) ' + distortion)
            ax.plot(np.round(100*fpr["micro"],2), np.round(100*tpr["micro"],2),
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
            a += 1

          fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
          roc_auc[i] = auc(fpr[i], tpr[i])

          color = colors[i % len(colors)]
          style = linestyles[i % len(linestyles)]
          marker = markers[i % len(markers)]
          ax.plot(np.round(100*fpr[i],2), np.round(100*tpr[i],2), linestyle=style, marker=marker, color=color, markersize=5,
                  label='class {0} (area = {1:0.2f})'.format(labels[i], roc_auc[i]));

          # only annotate the threshold for every 4 points in the true class
          if 't' in labels[i]:
            score = y_score[:, i]
            ti = thresholds[i]
            y_true = y_test[:, i]
            xi = np.round(100*fpr[i],2)
            yi = np.round(100*tpr[i],2)
            for k in range(0, len(ti), 2):
              y_pred = score >= ti[k]
              accuracy = accuracy_score(y_true, y_pred)
              ax.annotate('{0:.2f}, {1}%, {2}%, {3}%'.format(ti[k],np.round(100*accuracy,1),np.round(xi[k],1),np.round(yi[k],1)),
                            xy=(xi[k], yi[k]), xycoords='data',fontsize='xx-small',
                            xytext=(60, -30), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->"))
              if k == 8:
                ax.annotate('score, accuracy, FPR, TPR',
                            xy=(xi[k], yi[k]), xycoords='data', fontsize='xx-small',
                            xytext=(60, -40), textcoords='offset points')


          ax.legend(loc=0, fontsize='x-small')

        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir,  '_roc_all_by_class_' + distortion + '.png'), format='png',
                  dpi=120);
        plt.close('all')

    # confusion matrix plot
    matches = find_matches(model_out_dir,  'metrics_cm.csv')
    cmap = "YlGnBu"
    for cm_file in matches:
      df = pd.read_csv(cm_file, sep=',')
      if not df.empty:
        distortion = cm_file.split('/')[-2]
        df = pd.pivot_table(df,columns='predicted', values='num', index='actual')
        # reindex to align the row/columns as needed for a confusion matrix plot
        df.fillna(0, inplace=True)
        index = df.index.union(df.columns)
        df = df.reindex(index=index, columns=index, fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 10));
        ax.set_title('Confusion Matrix ' + distortion)
        sns.heatmap(df, cmap=cmap, vmax=30, annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=360)
        plt.tight_layout()
        plt.savefig(os.path.join(model_out_dir,  '_metrics_cm_by_class_' + distortion + '.png'), format='png', dpi=120);
        plt.close('all')


    plt.close('all')
  except Exception as ex:
    print('Error aggregating/plotting metrics {0}'.format(type(ex)))