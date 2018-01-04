#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Utility class refactored out of the TensorFlow code:
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
import pandas as pd
import re

from tensorflow.python.platform import gfile
from six.moves import urllib
from shutil import copyfile

from tensorflow.python.util import compat
import random
from tensorflow.python.framework import tensor_shape
import conf
from tensorflow.python.framework import ops
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score, average_precision_score
import xlsxwriter
import tempfile
import shutil

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def get_prediction_images(img_dir):
  """Grab images from the prediction directory.
  :param img_dir: the directory with images to predict
  :return: list of valid images
  """
  extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  file_list = []
  if not gfile.Exists(img_dir):
    print("Image directory '" + img_dir + "' not found.")
    return None
  print("Looking for prediction images in '" + img_dir + "'")
  for extension in extensions:
    file_glob = os.path.join(img_dir, '*.' + extension)
    file_list.extend(glob.glob(file_glob))
  if not file_list:
    print('No image files found')
    
  return file_list 

def make_image_predictions(sess, output_labels_file, bottleneck_dir, classifier, jpeg_data_tensor,
                           bottleneck_tensor, path_list, labels_list, target_dir, exemplars):
  """Use the learned model to make predictions."""

  if not labels_list:
    if gfile.Exists(output_labels_file):
      with open(output_labels_file, 'r') as lfile:
        labels_string = lfile.read()
        labels_list = json.loads(labels_string)
        print("labels list: %s" % labels_list)
    else:
      print("Labels list %s not found" % output_labels_file)
      return None

  bottlenecks = []
  print("Predicting for images: %s" % path_list)
  for img_path in path_list: 
    # get bottleneck for an image path
    dir, filename = os.path.split(img_path)
    bottleneck_path = os.path.join(bottleneck_dir, filename + '.txt')
    if not os.path.exists(bottleneck_path):
        print('Creating bottleneck at ' + bottleneck_path)
        if not gfile.Exists(img_path):
            tf.logging.fatal('File does not exist %s', img_path)
        image_data = gfile.FastGFile(img_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                    jpeg_data_tensor,
                                                    bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    bottlenecks.append(bottleneck_values)

  prediction_input = np.array(bottlenecks, dtype=np.float32)
  predictions = classifier.predict(x=prediction_input, as_iterable=True)
  print("Predictions:")
  workbook = xlsxwriter.Workbook(os.path.join(target_dir, 'predictions.xlsx'))
  temp_dir = tempfile.mkdtemp()
  with open(os.path.join(target_dir, 'predictions.csv'), "w") as f:
      f.write("Predicted,Filename,Score\n")
      row = 1
      worksheet = workbook.add_worksheet()
      worksheet.write(0, 0, 'Blob')
      worksheet.write(0, 1, 'Thumbnail')
      worksheet.write(0, 2, 'Predicted')
      worksheet.write(0, 3, 'Score')
      worksheet.write(0, 4, 'Exemplar')
      worksheet.write(0, 5, 'Filename')
      for i, p in enumerate(predictions):
        for k in p.keys():
          if k == "index":
            score = p['class_vector'][p[k]]
            label = labels_list[p[k]]
            if score > 0.50:
              print("index label is: %s score %f file: %s" % (label, score, path_list[i]))
              f.write("{0},{1},{2}\n".format(label, path_list[i], score))
              worksheet.set_row(row, height=50)
              worksheet.write(row, 2, label)
              worksheet.write(row, 3, score)
              try:
                  thumbnail_file = crop(temp_dir, random.choice(exemplars[label]))
                  worksheet.insert_image(row, 4, thumbnail_file)
                  thumbnail_file = crop(temp_dir, path_list[i])
                  worksheet.insert_image(row, 0, thumbnail_file)
                  fn = path_list[i].replace('/spectrogram/', '/spectrogram_raw/')
                  thumbnail_file = crop(temp_dir, fn, 'f')
                  worksheet.insert_image(row, 1, thumbnail_file)
              except Exception as ex:
                  print('{0}'.format(ex))
              worksheet.write(row, 5, path_list[i])
              row += 1

      # clean up
      workbook.close()
      shutil.rmtree(temp_dir)

def get_dims(image):
  """
  get the height and width of a tile
  :param image: the image file
  :return: height, width
  """
  cmd = 'identify "%s"' % (image)
  subproc = subprocess.Popen(cmd, env=os.environ, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  out, err = subproc.communicate()
  # get image height and width of raw tile
  p = re.compile(r'(?P<width>\d+)x(?P<height>\d+)')
  match = re.search(pattern=p, string=out)
  if (match):
      width = match.group("width")
      height = match.group("height")
      return height, width

  raise Exception('Cannot find height/width for image %s' % image)

def ensure_dir(d):
  """
  ensures a directory exists; creates it if it does not
  :param fname:
  :return:
  """
  if not os.path.exists(d):
    os.makedirs(d)

def maybe_download_and_extract(data_url, dest_dir='/tmp/imagenet'):
  """
  Download and extract model tar file.  If the pretrained model we're using doesn't already exist,
   downloads it and unpacks it into a directory.
  :param data_url:  url where tar.gz file exists
  :param dest_dir:  destination directory untar to
  :return:
  """
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_dir)


def write_list_of_floats_to_file(tensor_size, list_of_floats, file_path):
  """
   Writes a given list of floats to a binary file.
  :param tensor_size: size of the tensor of floats to write
  :param list_of_floats: List of floats we want to write to a file.
  :param file_path: Path to a file where list of floats will be stored.
  :return:
  """
  s = struct.pack('d' * tensor_size, *list_of_floats)
  with open(file_path, 'wb') as f:
    f.write(s)


def read_list_of_floats_from_file(tensor_size, file_path):
  """
   Reads list of floats from a given file.
  :param tensor_size: size of the tensor of floats to write
  :param file_path: Path to a file where list of floats was stored.
  :return:Array of bottleneck values (list of floats).
  """
  with open(file_path, 'rb') as f:
    s = struct.unpack('d' * tensor_size, f.read())
    return list(s)

def create_inception_graph(sess, model_filename):
    """"Creates a graph from saved GraphDef file and returns a Graph object.
  
    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
    """
    # import the graph and give me nodes where we want to pull the bottleneck data from
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
            tf.import_graph_def(graph_def, name='', return_elements=[
                conf.BOTTLENECK_TENSOR_NAME, conf.JPEG_DATA_TENSOR_NAME,
                conf.RESIZED_INPUT_TENSOR_NAME]))

    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def create_image_exemplars(image_dir):
  """Builds a list of examplar images
  Returns:
    A dictionary containing an entry for each example subfolder
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  exemplars = {}
  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for exemplar images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(glob.glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    example_images = []
    exemplars[label_name] = {}
    for file_name in file_list:
      example_images.append(file_name)
    exemplars[label_name] = example_images
  return exemplars

def create_image_lists(output_labels_file,
                       output_labels_file_lt20, image_dir,
                       testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stablestruct
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  If the model_dir already has a label list in it, use that to define the label
  ordering as the images are processed.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    output_labels_file: String path to a file where labels for this subfolder of image will be stored
    output_labels_lt20: String path to a file where labels with less than 20 images will be stored
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved
    for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None

  # See if the model dir contains an existing labels list. This will only be
  # the case if training using that model has occurred previously.
  labels_list = None
  if gfile.Exists(output_labels_file):
    with open(output_labels_file, 'r') as lfile:
      labels_string = lfile.read()
      labels_list = json.loads(labels_string)
      print("Found labels list: %s" % labels_list)

  labels_lt20 = []
  result = {}
  if labels_list:
    for l in labels_list:
      result[l] = {}

  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(glob.glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder {0} has less than 20 images, which may cause issues.'.format(dir_name))
      labels_lt20.append(dir_name.lower())
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {0} has more than {1} images. Some images will '
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file
      # should go into the training, testing, or validation sets, and we
      # want to keep existing files in the same set even if more files
      # are subsequently added.
      # To do that, we need a stable way of deciding based on just the
      # file name itself, so we do a hash of that and then use that to
      # generate a probability value that we use to assign it.
      if len(file_list) > 20:
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_IMAGES_PER_CLASS))
        if percentage_hash < validation_percentage:
          validation_images.append(base_name)
        elif percentage_hash < (testing_percentage + validation_percentage):
          testing_images.append(base_name)
        else:
          training_images.append(base_name)
      else:
          training_images.append(base_name)
          testing_images.append(base_name)
          validation_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
        'num_training_images': len(training_images),
    }

  labels = json.dumps(list(labels_lt20))
  with gfile.FastGFile(output_labels_file_lt20, 'w') as f:
    f.write(labels)

  return result

def get_all_cached_bottlenecks_multilabel_feedingtype(sess, df, image_lists, category, bottleneck_dir,
                               image_dir, jpeg_data_tensor, bottleneck_tensor):

  bottlenecks = []
  ground_truths = []

  label_names = list(image_lists.keys())

  # get a list of all labels by group and category(class); change cases to make them unique
  groups_unique = list(df.group.unique())
  groups = [name.lower() for name in groups_unique]
  feeding_types_unique = list(df[u'feeding.type'].unique())
  feeding_types = [name.upper() for name in feeding_types_unique]
  all_label_names = groups + feeding_types

  # go through the images in whatever order they are sorted - this might be by group or category(class)
  for label_index in range(len(label_names)):
    label_name = label_names[label_index]

    for image_index in range(len(image_lists[label_name][category])):

      bottleneck, image_path = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

      ground_truth = np.zeros((2, len(all_label_names)), dtype=np.float32)
      filename = os.path.split(image_path)[1]
      id = int(filename.split('.')[0])
      group = df.iloc[id].group
      feeding_type = df.iloc[id][u'feeding.type']
      ground_truth[0][all_label_names.index(group.lower())] = 1.0
      ground_truth[1][all_label_names.index(feeding_type.upper())] = 1.0
      ground_truths.append(ground_truth.flatten())
      bottlenecks.append(bottleneck)

  return bottlenecks, ground_truths, all_label_names


def get_all_cached_bottlenecks(sess, image_lists, category, bottleneck_dir,
                               image_dir, jpeg_data_tensor, bottleneck_tensor):

  bottlenecks = []
  ground_truths = []
  image_paths = []
  label_names = list(image_lists.keys())
  for label_index in range(len(label_names)):
    label_name = label_names[label_index]
    print('Label {0}'.format(label_name))

    for image_index in range(len(image_lists[label_name][category])):
      bottleneck, image_path = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
      ground_truth = np.zeros(len(label_names), dtype=np.float32)
      ground_truth[label_index] = 1.0
      ground_truths.append(ground_truth)
      bottlenecks.append(bottleneck)
      image_paths.append(image_path)

  return bottlenecks, ground_truths, image_paths

def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    bottleneck_tensor: The penultimate output layer of the graph.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_dir, category, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)
        how_many_bottlenecks += 1
        if how_many_bottlenecks % 2000 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')



def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from: training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    bottleneck_tensor: The output tensor for the bottleneck values.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
    Original image path string
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  image_path = get_image_path(image_lists, label_name, index, image_dir,
                                category)
  if not os.path.exists(bottleneck_path):
    print('Creating bottleneck at ' + bottleneck_path)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                jpeg_data_tensor,
                                                bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
      bottleneck_file.write(bottleneck_string)

  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()

  bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values, image_path

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The number of bottleneck values to return.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                          image_index, image_dir, category,
                                          bottleneck_dir, jpeg_data_tensor,
                                          bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_input_tensor,
                                         bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """

  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=conf.MODEL_INPUT_DEPTH)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, conf.MODEL_INPUT_WIDTH)
  precrop_height = tf.multiply(scale_value, conf.MODEL_INPUT_HEIGHT)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [conf.MODEL_INPUT_HEIGHT, conf.MODEL_INPUT_WIDTH,
                                  conf.MODEL_INPUT_DEPTH])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def save_metrics(args, classifier, bottlenecks, all_label_names, test_ground_truth, image_paths, image_lists, exemplars):

  sess = tf.Session()
  with sess.as_default():
    results_y_test = {}
    results_y_score = {}
    df = pd.DataFrame(columns=['actual','predicted','num'])
    df_roc = pd.DataFrame(columns=['y_test', 'y_score', 'labels'], index=range(test_ground_truth.shape[0]))

    predictions = classifier.predict(x=bottlenecks, as_iterable=True)

    for key in all_label_names:
      results_y_test[key] = []
      results_y_score[key] = []

    y_true = np.empty(test_ground_truth.shape[0])
    y_pred = np.empty(test_ground_truth.shape[0])

    workbook = xlsxwriter.Workbook(os.path.join(args.model_dir, 'misclassified.xlsx'))
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(args.model_dir,'misclassified.csv'), "w") as f2:
        f2.write("Actual,Predicted,Filename\n")
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'Thumbnail')
        worksheet.write(0, 1, 'Actual')
        worksheet.write(0, 2, 'Predicted')
        worksheet.write(0, 3, 'Score')
        worksheet.write(0, 4, 'Exemplar')
        worksheet.write(0, 5, 'Filename')

        # calculate the scores for predictions as needed for scipy functions
        row = 1
        for j, p in enumerate(predictions):
            print("---------")
            predicted = int(p['index'])
            actual = int(np.argmax(test_ground_truth[j]))
            y_true[j] = actual
            y_pred[j] = predicted
            df_roc.loc[j] = {'y_test': test_ground_truth[j], 'y_score': p['class_vector'], 'labels': all_label_names}

            print("%i is predicted as %s actual class %s %i %i" % (j, all_label_names[predicted], all_label_names[actual],\
                                                                   predicted, actual))
            if df.ix[(df.actual == all_label_names[actual]) & (df.predicted == all_label_names[predicted])].empty:
                df = df.append([{'actual': all_label_names[actual], 'predicted': all_label_names[predicted], 'num': 0}])
            df.ix[(df.actual == all_label_names[actual]) & (df.predicted == all_label_names[predicted]), 'num'] += 1

            if predicted is not actual:
                f2.write("{0},{1},{2}\n".format(all_label_names[actual], all_label_names[predicted], image_paths[j]))
                thumbnail_file = crop(temp_dir, image_paths[j])
                worksheet.set_row(row, height=50)
                worksheet.insert_image(row, 0, thumbnail_file)
                worksheet.write(row, 1, all_label_names[actual])
                worksheet.write(row, 2, all_label_names[predicted])
                worksheet.write(row, 3,  p['class_vector'][predicted])
                try:
                    thumbnail_file = crop(temp_dir, random.choice(exemplars[all_label_names[predicted]]))
                    worksheet.insert_image(row, 4, thumbnail_file)
                except Exception as ex:
                    print('{0}'.format(ex))
                worksheet.write(row, 5, image_paths[j])
                row += 1

    workbook.close()
    shutil.rmtree(temp_dir)
    accuracy_all = accuracy_score(y_true, y_pred)
    precision_all = precision_score(y_true, y_pred)
    f1_all = f1_score(y_true, y_pred)

    df_roc.to_pickle(os.path.join(args.model_dir, 'metrics_roc.pkl'))
    distortion = 'None'
    with open(os.path.join(args.model_dir,'metrics.csv'), "w") as f:
        f.write("Distortion,Accuracy,Precision,F1\n")
        if args.random_crop:
          distortion = "{0}_{1:2d}".format("random_crop", int(args.random_crop))
        if args.random_scale:
          distortion = "{0}_{1:2d}".format("random_scale", int(args.random_scale))
        if args.random_brightness:
          distortion = "{0}_{1:2d}".format("random_brightness", int(args.random_brightness))
        f.write("{0},{1:1.5f},{2:1.5f},{3:1.5f}\n".format(distortion, accuracy_all, precision_all, f1_all))

        ind = np.arange(len(all_label_names))  # the x locations for the classes
        precision = precision_score(y_true, y_pred, labels=ind, average=None)
        recall = recall_score(y_true, y_pred, labels=ind, average=None)
        f1 = f1_score(y_true, y_pred, labels=ind, average=None)

    with open(os.path.join(args.model_dir,'metrics_by_class.csv'), "w") as f:
        f.write("Distortion,Class,NumTrainingImages,Accuracy,Precision,F1\n")
        for i in range(len(recall)):
          class_name = all_label_names[i]
          f.write("{0},{1},{2},{3:1.5f},{4:1.5f},{5:1.5f}\n".format(distortion, class_name, image_lists[class_name]['num_training_images'], recall[i], precision[i], f1[i]))

    # save CM as a csv file
    with open(os.path.join(args.model_dir,'metrics_cm.csv'), "w") as f:
        f.write(','.join(all_label_names) + '\n')
        for i in range(len(all_label_names)):
          class_name = all_label_names[i]
          f.write("{0},{1},{2},{3:1.5f},{4:1.5f},{5:1.5f}\n".format(distortion, class_name, image_lists[class_name]['num_training_images'], recall[i], precision[i], f1[i]))

    df.to_csv(os.path.join(args.model_dir,'metrics_cm.csv'), float_format='%1.5f')
    print('Done')


def crop(temp_dir, image_path, prepend_str=""):
    path, filename = os.path.split(image_path)
    thumbnail_file = os.path.join(temp_dir, prepend_str + 'thumb_' + filename)
    os.system('/usr/local/bin/convert "%s" -thumbnail 50x50 -unsharp 0x.5 "%s"' % (image_path, thumbnail_file))
    return thumbnail_file
