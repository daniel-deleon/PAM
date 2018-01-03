#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2016, MBARI"
__license__ = "GNU License"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__status__ = "Development"
__doc__ = '''

This script runs transfer learning on the data using the Inception v3 model trained on ImageNet
Produces performance metrics for evaluating using sklearn

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
import os
import numpy as np
import util
import util_plot
import time
import transfer_model as transfer_model
import tensorflow as tf
from tensorflow.python.platform import gfile


def process_command_line():
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += sys.argv[0] + " --image_dir /tmp/cropped_images/" \
                              " --bottleneck_dir /tmp/cropped_images/bottleneck" \
                              " --model_dir /tmp/model_output/default"
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Run transfer learning on folder of images organized by label ',
                                     epilog=examples)

    # Input and output file flags.
    parser.add_argument('--image_dir', type=str, required=True,  help="Path to folders of labeled images.")
    parser.add_argument('--exemplar_dir', type=str, required=True,  help="Path to folders of exemplar images for each label")

    # where the model information lives
    parser.add_argument('--model_dir', type=str, default=os.path.join( "/tmp/tfmodels/img_classify", str(int(time.time()))), help='Directory for storing model info')

    # Details of the training configuration.
    parser.add_argument('--num_steps', type=int, default=15000, help="How many training steps to run before ending.")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="How large a learning rate to use when training.")
    parser.add_argument('--testing_percentage', type=int, default=10, help="What percentage of images to use as a test set.")
    parser.add_argument('--validation_percentage', type=int, default=10, help="What percentage of images to use as a validation set.")
    parser.add_argument('--eval_step_interval', type=int, default=10, help="How often to evaluate the training results.")
    parser.add_argument('--train_batch_size', type=int, default=100, help="How many images to train on at a time.")
    parser.add_argument('--test_batch_size', type=int, default=500,
                        help="""How many images to test on at a time. This
                        test set is only used infrequently to verify
                        the overall accuracy of the model.""")
    parser.add_argument( '--validation_batch_size', type=int, default=100,
                        help="""How many images to use in an evaluation batch. This validation
                        set is used much more often than the test set, and is an early
                        indicator of how accurate the model is during training.""")

    # File-system cache locations.
    parser.add_argument('--incp_model_dir', type=str, default='/tmp/imagenet', help="""Path to graph.pb for a given model""")
    parser.add_argument('--bottleneck_dir', type=str, default='/tmp/bottlenecks', help="Path to cache bottleneck layer values as files.")
    parser.add_argument('--final_tensor_name', type=str, default='final_result', help="The name of the output classification layer in the retrained graph.")

    # Controls the distortions used during training.
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--flip_left_right', action='store_true', default=False, help="Whether to randomly flip the training images horizontally.")
    parser.add_argument('--random_crop', type=int, default=0, help="""A percentage determining how much of a margin to randomly crop off the training images.""")
    parser.add_argument('--random_scale', type=int, default=0, help="""A percentage determining how much to randomly scale up the size of the training images by.""")
    parser.add_argument('--random_brightness', type=int, default=0, help="""A percentage determining how much to randomly multiply the training image input pixels up or down by.""")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = process_command_line()
    sess = tf.Session()

    # Set up the pre-trained graph.
    print("Using model directory {0} and model from {1}".format(args.model_dir, conf.DATA_URL))
    '''util.ensure_dir(args.model_dir)
util.maybe_download_and_extract(data_url=conf.DATA_URL, dest_dir=args.incp_model_dir)
model_filename = os.path.join(args.incp_model_dir, conf.MODEL_GRAPH_NAME)
graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor=(util.create_inception_graph(sess, model_filename))

labels_list = None
output_labels_file = os.path.join(args.model_dir, "output_labels.json")
output_labels_file_lt20 = os.path.join(args.model_dir, "output_labels_lt20.json")
d = os.path.dirname(output_labels_file_lt20)
util.ensure_dir(d)

# Look at the folder structure, and create lists of all the images.
image_lists = util.create_image_lists(output_labels_file,
                                      output_labels_file_lt20,
                                      args.image_dir, args.testing_percentage,
                                      args.validation_percentage)

class_count = len(image_lists.keys())
if class_count == 0:
  print('No valid folders of images found at ' + args.image_dir)
  exit(-1)
if class_count == 1:
  print('Only one valid folder of images found at ' + args.image_dir +
        ' - multiple classes are needed for classification.')
  exit(-1)

# See if the command-line flags mean we're applying any distortions.
do_distort_images =  (args.flip_left_right or (args.random_crop != 0)
                      or (args.random_scale != 0) or
                      (args.random_brightness != 0))
sess = tf.Session()

# Create example images
exemplars = util.create_image_exemplars(args.exemplar_dir)

if do_distort_images:
# We will be applying distortions, so setup the operations we'll need.
distorted_jpeg_data_tensor, distorted_image_tensor = util.add_input_distortions(
  args.flip_left_right, args.random_crop, args.random_scale,
  args.random_brightness)
else:
# We'll make sure we've calculated the 'bottleneck' image summaries and
# cached them on disk.
util.cache_bottlenecks(sess, image_lists, args.image_dir, args.bottleneck_dir,
                jpeg_data_tensor, bottleneck_tensor)

all_label_names = list(image_lists.keys())
train_bottlenecks, train_ground_truth, image_paths = util.get_all_cached_bottlenecks(
                                                                sess,
                                                                image_lists, 'training',
                                                                args.bottleneck_dir, args.image_dir,
                                                                jpeg_data_tensor, bottleneck_tensor)
train_bottlenecks = np.array(train_bottlenecks)
train_ground_truth = np.array(train_ground_truth)

# Define the custom estimator
model_fn = transfer_model.make_model_fn(class_count, args.final_tensor_name, args.learning_rate)

model_params = {}
classifier = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, model_dir=args.model_dir)

# run the training
print("Starting training for %s steps max" % args.num_steps)
classifier.fit(
x=train_bottlenecks.astype(np.float32),
y=train_ground_truth, batch_size=10,
max_steps=args.num_steps)

# We've completed our training, so run a test evaluation on some new images we haven't used before.
test_bottlenecks, test_ground_truth, image_paths = util.get_all_cached_bottlenecks(
                                              sess, image_lists, 'testing',
                                              args.bottleneck_dir, args.image_dir, jpeg_data_tensor,
                                              bottleneck_tensor)
test_bottlenecks = np.array(test_bottlenecks)
test_ground_truth = np.array(test_ground_truth)
print("evaluating....")
classifier.evaluate(test_bottlenecks.astype(np.float32), test_ground_truth)

# write the output labels file if it doesn't already exist
if gfile.Exists(output_labels_file):
print("Labels list file already exists; not writing.")
else:
output_labels = json.dumps(list(image_lists.keys()))
with gfile.FastGFile(output_labels_file, 'w') as f:
f.write(output_labels)

print("\nSaving metrics...")
util.save_metrics(args, classifier, test_bottlenecks.astype(np.float32), all_label_names, test_ground_truth, \
          image_paths, image_lists, exemplars)'''

    util_plot.plot_metrics(args.model_dir)

    print("Done !")