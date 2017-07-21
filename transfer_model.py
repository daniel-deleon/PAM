#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Model class refactored out of the TensorFlow code:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import glob
import hashlib
import json
import numpy as np
import os
import sys
import struct
import subprocess
import tarfile
import tensorflow as tf
import re
import conf

from tensorflow.python.platform import gfile
from six.moves import urllib

import tensorflow as tf
from tensorflow.contrib.metrics.python.ops import metric_ops
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn import ModeKeys


def make_model_fn(class_count, final_tensor_name, learning_rate):

  def _make_model(bottleneck_input, ground_truth_input, mode, params):

    prediction_dict = {}
    train_step = None
    cross_entropy = None

    # Add the new layer that we'll be training.
    (train_step, cross_entropy,
     final_tensor) = add_final_training_ops(learning_rate,
        class_count, mode, final_tensor_name,
        bottleneck_input, ground_truth_input)

    if mode == ModeKeys.EVAL:
      prediction_dict['loss'] = cross_entropy
      # Create the operations we need to evaluate accuracy
      acc = add_evaluation_step(final_tensor, ground_truth_input)
      prediction_dict['accuracy'] = acc

    if mode == ModeKeys.INFER:
      predclass = tf.argmax(final_tensor, 1)
      prediction_dict["class_vector"] = final_tensor
      prediction_dict["index"] = predclass

    return prediction_dict, cross_entropy, train_step

  return _make_model


def add_final_training_ops(learning_rate,
    class_count, mode, final_tensor_name,
    bottleneck_input, ground_truth_input):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this
  function adds the right operations to the graph, along with some variables
  to hold the weights, and then sets up all the gradients for the backward
  pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    learning_rate: learning rate
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces
    results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  train_step = None
  cross_entropy_mean = None

  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      layer_weights = tf.Variable(
          tf.truncated_normal(
              [conf.BOTTLENECK_TENSOR_SIZE, class_count],
              stddev=0.001), name='final_weights')
      variable_summaries(layer_weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram(layer_name + '/pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram(final_tensor_name + '/activations', final_tensor)

  if mode in [ModeKeys.EVAL, ModeKeys.TRAIN]:
    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=ground_truth_input)
      with tf.name_scope('total'):
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
      tf.summary.scalar('cross entropy', cross_entropy_mean)

    with tf.name_scope('train'):
      train_step = tf.train.GradientDescentOptimizer(
          learning_rate).minimize(
              cross_entropy_mean,
              global_step=tf.contrib.framework.get_global_step())

  return (train_step, cross_entropy_mean, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
  """
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(result_tensor, 1), \
                                    tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

