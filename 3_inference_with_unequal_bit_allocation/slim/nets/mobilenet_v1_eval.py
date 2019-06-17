# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validate mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import mobilenet_v1
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to distinguish')
flags.DEFINE_integer('num_examples', 50000, 'Number of examples to evaluate')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir', '', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '', 'Location of dataset')
# quantization command line parameters
flags.DEFINE_string('bit_allocation', '', 'the filename of bit allocation file')
flags.DEFINE_string('dir_weight_codebooks', '', 'path of codebooks of weights')
flags.DEFINE_string('dir_activation_codebooks', '', 'path of codebooks of activations')

FLAGS = flags.FLAGS

inf = 100000000
num_quant_layer_wei = 28
num_quant_layer_act = 27
wei_quant_lambda = []
wei_quant_delta = []
wei_quant_levels = []
act_quant_delta = []
act_quant_levels = []

def read_all_lines(filename):
	with open(filename) as f:
		data = f.readlines()
	return data


def load_bits_allocation(input_file):
	dead_zones = []
	quant_levels = []

	with open(input_file) as f:
		for line in f:
			a, b = [int(x) for x in line.split()]
			dead_zones.append(a)
			quant_levels.append(b)

	return dead_zones, quant_levels


def load_codebook_wei(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []

    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[0], cluster[1]


def load_codebook_act(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []

    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[0]

def load_all_weights_name():
	data = read_all_lines('list_mobilenet_v1_weights.txt')
	all_weights_names = []

	for eachline in data[0:]:
		all_weights_names.append(eachline.strip('\n'))

	return all_weights_names


def load_quantization_configs(dir_bit_allocation = [], path_wei_codebooks = [], path_act_codebooks = []):
	all_weights_names = load_all_weights_name()
	dead_zones, quant_levels = load_bits_allocation(dir_bit_allocation)
	
	for i in range(num_quant_layer_wei):
		dir_wei_codebooks = path_wei_codebooks + '/' + all_weights_names[i] + '_' + str(dead_zones[i]) + '_' + str(quant_levels[i]) + '.cb'
		quant_lambda, quant_delta = load_codebook_wei(dir_wei_codebooks)
		wei_quant_lambda.append(quant_lambda)
		wei_quant_delta.append(quant_delta)
		wei_quant_levels.append(quant_levels[i])

	for i in range(num_quant_layer_act):
		dir_act_codebooks = path_act_codebooks + '/' + 'act_layer_' + str(i) + '.dat_' + str(quant_levels[i + num_quant_layer_wei]) + '.cb'
		if inf == quant_levels[i + num_quant_layer_wei]:
			quant_delta = -1
		else:
			quant_delta = load_codebook_act(dir_act_codebooks)
		act_quant_delta.append(quant_delta)
		act_quant_levels.append(quant_levels[i + num_quant_layer_wei])

	return



def imagenet_input(is_training):
  """Data reader for imagenet.

  Reads in imagenet data and performs pre-processing on the images.

  Args:
     is_training: bool specifying if train or validation dataset is needed.
  Returns:
     A batch of images and labels.
  """
  if is_training:
    dataset = dataset_factory.get_dataset('imagenet', 'train',
                                          FLAGS.dataset_dir)
  else:
    dataset = dataset_factory.get_dataset('imagenet', 'validation',
                                          FLAGS.dataset_dir)

  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2 * FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])

  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      'mobilenet_v1', is_training=is_training)

  image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

  images, labels = tf.train.batch(
      tensors=[image, label],
      batch_size=FLAGS.batch_size,
      num_threads=4,
      capacity=5 * FLAGS.batch_size)
  return images, labels


def metrics(logits, labels):
  """Specify the metrics for eval.

  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.

  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), labels),
      'Recall_5': tf.metrics.recall_at_k(labels, logits, 5),
  })
  #for name, value in names_to_values.iteritems():
  for name, value in names_to_values.items():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return list(names_to_updates.values())


def build_model():
  """Build the mobilenet_v1 model for evaluation.

  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    inputs, labels = imagenet_input(is_training=False)

    scope = mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)
    with slim.arg_scope(scope):
      logits, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes,
          wei_quant_lambda_input = wei_quant_lambda, wei_quant_delta_input = wei_quant_delta, wei_quant_levels_input = wei_quant_levels, act_quant_delta_input = act_quant_delta, act_quant_levels_input = act_quant_levels)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    eval_ops = metrics(logits, labels)

  return g, eval_ops


def eval_model():
  """Evaluates mobilenet_v1."""
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    slim.evaluation.evaluate_once(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops)


def main(unused_arg):
    load_quantization_configs(FLAGS.bit_allocation, FLAGS.dir_weight_codebooks, FLAGS.dir_activation_codebooks)

    for i in range(len(wei_quant_lambda)):
        print('wei: %f %f %f' % (wei_quant_lambda[i], wei_quant_delta[i], wei_quant_levels[i]))

    for i in range(len(act_quant_delta)):
        print('act %f %f' % (act_quant_delta[i], act_quant_levels[i]))

    eval_model()

if __name__ == '__main__':
  tf.app.run(main)
