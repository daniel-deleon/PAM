#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2017, MBARI"
__license__ = "GNU License"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__status__ = "Development"
__doc__ = '''

This script runs transfer learning on the data using the Inception v3 model trained on ImageNet

Based on the TensorFlow code:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

Prerequisites:

@undocumented: __doc__ parser
@author: __author__
@status: __status__
@license: __license__
'''

import json
import conf
import sys
import argparse
import util
import time
import transfer_model as transfer_model
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import tarfile
import util
import tempfile
import shutil

def process_command_line():
  from argparse import RawTextHelpFormatter

  examples = 'Examples:' + '\n\n'
  examples += sys.argv[0] + " --bottleneck_dir /tmp/bottleneck/" \
                            " --model_dir /tmp/model_output/random_scale_20/"
  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                   description='Run transfer learning on BLED detections ',
                                   epilog=examples)

  # Input and output file flags.
  parser.add_argument('--input_dir', type=str, required=True, help="Directory of images to predict")
  parser.add_argument('--prediction_dir', type=str, required=True, help="Directory to store predicted output images")

  # where the model information lives
  parser.add_argument('--model_dir', type=str, required=True,
                      default=os.path.join(os.getcwd() + '/model', str(int(time.time()))),
                      help='Directory for storing model info')
  # Details of the training configuration.
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help="How large a learning rate to use when training.")

  # File-system cache locations.
  parser.add_argument('--incp_model_dir', type=str, default='/tmp/imagenet',
                      help="""Path to graph.pb for a given model""")
  parser.add_argument('--final_tensor_name', type=str, default='final_result',
                      help="The name of the output classification layer in the retrained graph.")
  parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottlenecks',
                      help="Path to cache bottleneck layer values as files.")

  args = parser.parse_args()
  return args


if __name__ == '__main__':

    args = process_command_line()
    sess = tf.Session()

    if not os.path.exists(args.input_dir):
      print ('Need to specify input directory with the --input_dir option')

    if not os.path.exists(args.prediction_dir):
      print('Need to specify directory to store predictions with the --prediction_dir option')

          # Set up the pre-trained graph.
    print("Using model directory {0} and model from {1}".format(args.model_dir, conf.DATA_URL))
    util.ensure_dir(args.model_dir)
    util.maybe_download_and_extract(data_url=conf.DATA_URL, dest_dir=args.incp_model_dir)
    model_filename = os.path.join(args.incp_model_dir, conf.MODEL_GRAPH_NAME)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (util.create_inception_graph(sess, model_filename))

    labels_list = None
    output_labels_file = os.path.join(args.model_dir, "output_labels.json")
    output_labels_file_lt20 = os.path.join(args.model_dir, "output_labels_lt20.json")
    d = os.path.dirname(output_labels_file_lt20)
    util.ensure_dir(d)

    util.ensure_dir(args.bottleneck_dir)

    # load the labels list, needed to create the model; exit if it's not there
    if gfile.Exists(output_labels_file):
      with open(output_labels_file, 'r') as lfile:
        labels_string = lfile.read()
        labels_list = json.loads(labels_string)
        print("labels list: %s" % labels_list)
        class_count = len(labels_list)
    else:
        print("Labels list %s not found" % output_labels_file)
        exit(-1)

    # Define the custom estimator
    model_fn = transfer_model.make_model_fn(class_count, args.final_tensor_name, args.learning_rate)

    model_params = {}
    classifier = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, model_dir=args.model_dir)

    # run bottleneck computation and classification
    # Get prediction image list
    img_list = util.get_prediction_images(args.input_dir)

    print("Predicting...")
    if not img_list:
        print("No images found in %s" % args.input_dir)
    else:
        util.ensure_dir(args.prediction_dir)
        util.make_image_predictions(sess, output_labels_file, args.bottleneck_dir, classifier, jpeg_data_tensor, bottleneck_tensor,
                        img_list, labels_list, args.prediction_dir)

    print("Done !")
