# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def mapping_to_codebook(x, codebook):
  max_number_processings = 500000

  mat_size = x.get_shape().as_list()
  codebook_size = codebook.get_shape().as_list()
  len_x = np.prod(mat_size)
  len_c = np.prod(codebook_size)
  x_flat = tf.reshape(x, [-1])
#  x_temp = tf.expand_dims(x_flat, axis=1)
  c_temp = tf.expand_dims(codebook, axis=0)
  
  len_number_processings = len_x
  if len_number_processings > max_number_processings:
    len_number_processings = max_number_processings

  id_processing = 0

#  x_temp = tf.tile(x_temp, [1, len_c])
  c_temp = tf.tile(c_temp, [len_number_processings, 1])

  range_x = tf.range(len_number_processings, dtype=tf.int32)

  has_results = 0

  while True:
    id_processing_end = id_processing + len_number_processings - 1
    is_break = 0
    
    if id_processing_end + 1 >= len_x:
      is_break = 1
      id_processing_end = len_x - 1
      c_temp = tf.slice(c_temp , [0, 0], [id_processing_end - id_processing + 1 , len_c])
      range_x = tf.range(id_processing_end - id_processing + 1, dtype=tf.int32)
    
    x_temp = tf.expand_dims(tf.slice(x_flat, [id_processing], [id_processing_end - id_processing + 1]), axis=1)
    #x_temp_processing = tf.tile(x_temp , [1, len_c])

    d = tf.abs(tf.tile(x_temp , [1, len_c]) - c_temp)
    select_id = tf.argmin(d, axis=1, output_type=tf.int32)
    
    if has_results == 0:
      new_x = tf.gather_nd(c_temp, tf.stack((range_x, select_id),-1))
      has_results = 1
    else:
      ttemp_new_x = tf.gather_nd(c_temp, tf.stack((range_x, select_id),-1))
      new_x = tf.concat([new_x , ttemp_new_x] , 0)

    if is_break == 1:
      break

    id_processing = id_processing_end + 1

  new_x = tf.reshape(new_x, mat_size)

  return new_x 

def mapping_to_codebook_scalar_quant(x, st, base, num_levels):

  ts_st = tf.constant(st , dtype=tf.float32)
  ts_base = tf.constant(base , dtype=tf.float32)
  ts_num_levels = tf.constant(1.0 * (num_levels - 1) , dtype=tf.float32)

  x = tf.cast(x , dtype=tf.float32)

  d = tf.subtract(x , ts_st)
  e = tf.divide(d , ts_base)
  f = tf.round(e)
  g = tf.clip_by_value(f , clip_value_min = tf.constant(0.0), clip_value_max = ts_num_levels)
  h = tf.multiply(g , ts_base)
  
  result = tf.add(h , ts_st)  

  return result

def mapping_to_codebook_original(x, codebook):
  mat_size = x.get_shape().as_list()
  codebook_size = codebook.get_shape().as_list()
  len_x = np.prod(mat_size)
  len_c = np.prod(codebook_size)
  x_flat = tf.reshape(x, [-1])
  x_temp = tf.expand_dims(x_flat, axis=1)
  c_temp = tf.expand_dims(codebook, axis=0)
  x_temp = tf.tile(x_temp, [1, len_c])
  c_temp = tf.tile(c_temp, [len_x, 1])
  d = tf.abs(x_temp - c_temp)
  select_id = tf.argmin(d, axis=1)
  range_x = tf.range(len_x)
  new_x = tf.gather_nd(c_temp, tf.stack((range_x, select_id),-1))
  new_x = tf.reshape(new_x, mat_size)

  return new_x 


def mapping_to_codebook_divided_memory_reduced(x, codebook):
  max_number_processings = 500000

  mat_size = x.get_shape().as_list()
  codebook_size = codebook.get_shape().as_list()
  len_x = np.prod(mat_size)
  len_c = np.prod(codebook_size)
  x_flat = tf.reshape(x, [-1])
#  x_temp = tf.expand_dims(x_flat, axis=1)
  c_temp = tf.expand_dims(codebook, axis=0)
  
  len_number_processings = len_x
  if len_number_processings > max_number_processings:
    len_number_processings = max_number_processings

  id_processing = 0

#  x_temp = tf.tile(x_temp, [1, len_c])
  c_temp = tf.tile(c_temp, [len_number_processings, 1])

  range_x = tf.range(len_number_processings, dtype=tf.int32)

  has_results = 0

  while True:
    id_processing_end = id_processing + len_number_processings - 1
    is_break = 0
    
    if id_processing_end + 1 >= len_x:
      is_break = 1
      id_processing_end = len_x - 1
      c_temp = tf.slice(c_temp , [0, 0], [id_processing_end - id_processing + 1 , len_c])
      range_x = tf.range(id_processing_end - id_processing + 1, dtype=tf.int32)
    
    x_temp = tf.expand_dims(tf.slice(x_flat, [id_processing], [id_processing_end - id_processing + 1]), axis=1)
    x_temp_processing = tf.tile(x_temp , [1, len_c])

    d = tf.abs(x_temp_processing - c_temp)
    select_id = tf.argmin(d, axis=1, output_type=tf.int32)
    
    if has_results == 0:
      new_x = tf.gather_nd(c_temp, tf.stack((range_x, select_id),-1))
      has_results = 1
    else:
      ttemp_new_x = tf.gather_nd(c_temp, tf.stack((range_x, select_id),-1))
      new_x = tf.concat([new_x , ttemp_new_x] , 0)

    if is_break == 1:
      break

    id_processing = id_processing_end + 1

  new_x = tf.reshape(new_x, mat_size)

  return new_x 

def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')
      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')

      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')

      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_16_original(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224

def vgg_16_quant_act(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           layer_quantized=-1):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')
      if layer_quantized == 1:
        codebook_conv_1 = tf.get_variable(name='conv1/wz/codebook', shape=[512,])
        net_conv1 = mapping_to_codebook(net_conv1, codebook_conv_1)

      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')
      if layer_quantized == 2:
        codebook_conv_2 = tf.get_variable(name='conv2/wz/codebook', shape=[512,])
        net_conv2 = mapping_to_codebook(net_conv2, codebook_conv_2)

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      if layer_quantized == 3:
        codebook_conv_3 = tf.get_variable(name='conv3/wz/codebook', shape=[512,])
        net_conv3 = mapping_to_codebook(net_conv3, codebook_conv_3)

      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')
      if layer_quantized == 4:
        codebook_conv_4 = tf.get_variable(name='conv4/wz/codebook', shape=[512,])
        net_conv4 = mapping_to_codebook(net_conv4, codebook_conv_4)

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      if layer_quantized == 5:
        codebook_conv_5 = tf.get_variable(name='conv5/wz/codebook', shape=[512,])
        net_conv5 = mapping_to_codebook(net_conv5, codebook_conv_5)
	  
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      if layer_quantized == 6:
        codebook_conv_6 = tf.get_variable(name='conv6/wz/codebook', shape=[512,])
        net_conv6 = mapping_to_codebook(net_conv6, codebook_conv_6)

      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')
      if layer_quantized == 7:
        codebook_conv_7 = tf.get_variable(name='conv7/wz/codebook', shape=[512,])
        net_conv7 = mapping_to_codebook(net_conv7, codebook_conv_7)


      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      if layer_quantized == 8:
        codebook_conv_8 = tf.get_variable(name='conv8/wz/codebook', shape=[512,])
        net_conv8 = mapping_to_codebook(net_conv8, codebook_conv_8)

      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      if layer_quantized == 9:
        codebook_conv_9 = tf.get_variable(name='conv9/wz/codebook', shape=[512,])
        net_conv9 = mapping_to_codebook(net_conv9, codebook_conv_9)

      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')
      if layer_quantized == 10:
        codebook_conv_10 = tf.get_variable(name='conv10/wz/codebook', shape=[512,])
        net_conv10 = mapping_to_codebook(net_conv10, codebook_conv_10)


      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      if layer_quantized == 11:
        codebook_conv_11 = tf.get_variable(name='conv11/wz/codebook', shape=[512,])
        net_conv11 = mapping_to_codebook(net_conv11, codebook_conv_11)

      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      if layer_quantized == 12:
        codebook_conv_12 = tf.get_variable(name='conv12/wz/codebook', shape=[512,])
        net_conv12 = mapping_to_codebook(net_conv12, codebook_conv_12)

      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')
      if layer_quantized == 13:
        codebook_conv_13 = tf.get_variable(name='conv13/wz/codebook', shape=[512,])
        net_conv13 = mapping_to_codebook(net_conv13, codebook_conv_13)

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      if layer_quantized == 14:
        codebook_conv_14 = tf.get_variable(name='conv14/wz/codebook', shape=[512,])
        net_conv14 = mapping_to_codebook(net_conv14, codebook_conv_14)

      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      if layer_quantized == 15:
        codebook_conv_15 = tf.get_variable(name='conv15/wz/codebook', shape=[512,])
        net_conv15 = mapping_to_codebook(net_conv15, codebook_conv_15)

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15

      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net

      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        if layer_quantized == 16:
          codebook_conv_16 = tf.get_variable(name='conv16/wz/codebook', shape=[512,])
          net_conv16 = mapping_to_codebook(net_conv16, codebook_conv_16)


        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_16_quant_act_scalar(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           layer_quantized=-1,
           st=-1,
           base=-1,
           num_levels=-1):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')
      if layer_quantized == 1:
        #codebook_conv_1 = tf.get_variable(name='conv1/wz/codebook', shape=[512,])
        net_conv1 = mapping_to_codebook_scalar_quant(net_conv1, st, base, num_levels)

      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')
      if layer_quantized == 2:
        #codebook_conv_2 = tf.get_variable(name='conv2/wz/codebook', shape=[512,])
        net_conv2 = mapping_to_codebook_scalar_quant(net_conv2, st, base, num_levels)

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      if layer_quantized == 3:
        #codebook_conv_3 = tf.get_variable(name='conv3/wz/codebook', shape=[512,])
        net_conv3 = mapping_to_codebook_scalar_quant(net_conv3, st, base, num_levels)

      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')
      if layer_quantized == 4:
        #codebook_conv_4 = tf.get_variable(name='conv4/wz/codebook', shape=[512,])
        net_conv4 = mapping_to_codebook_scalar_quant(net_conv4, st, base, num_levels)

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      if layer_quantized == 5:
        #codebook_conv_5 = tf.get_variable(name='conv5/wz/codebook', shape=[512,])
        net_conv5 = mapping_to_codebook_scalar_quant(net_conv5, st, base, num_levels)
	  
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      if layer_quantized == 6:
        #codebook_conv_6 = tf.get_variable(name='conv6/wz/codebook', shape=[512,])
        net_conv6 = mapping_to_codebook_scalar_quant(net_conv6, st, base, num_levels)

      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')
      if layer_quantized == 7:
        #codebook_conv_7 = tf.get_variable(name='conv7/wz/codebook', shape=[512,])
        net_conv7 = mapping_to_codebook_scalar_quant(net_conv7, st, base, num_levels)


      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      if layer_quantized == 8:
        #codebook_conv_8 = tf.get_variable(name='conv8/wz/codebook', shape=[512,])
        net_conv8 = mapping_to_codebook_scalar_quant(net_conv8, st, base, num_levels)

      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      if layer_quantized == 9:
        #codebook_conv_9 = tf.get_variable(name='conv9/wz/codebook', shape=[512,])
        net_conv9 = mapping_to_codebook_scalar_quant(net_conv9, st, base, num_levels)

      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')
      if layer_quantized == 10:
        #codebook_conv_10 = tf.get_variable(name='conv10/wz/codebook', shape=[512,])
        net_conv10 = mapping_to_codebook_scalar_quant(net_conv10, st, base, num_levels)


      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      if layer_quantized == 11:
        #codebook_conv_11 = tf.get_variable(name='conv11/wz/codebook', shape=[512,])
        net_conv11 = mapping_to_codebook_scalar_quant(net_conv11, st, base, num_levels)

      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      if layer_quantized == 12:
        #codebook_conv_12 = tf.get_variable(name='conv12/wz/codebook', shape=[512,])
        net_conv12 = mapping_to_codebook_scalar_quant(net_conv12, st, base, num_levels)

      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')
      if layer_quantized == 13:
        #codebook_conv_13 = tf.get_variable(name='conv13/wz/codebook', shape=[512,])
        net_conv13 = mapping_to_codebook_scalar_quant(net_conv13, st, base, num_levels)

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      if layer_quantized == 14:
        #codebook_conv_14 = tf.get_variable(name='conv14/wz/codebook', shape=[512,])
        net_conv14 = mapping_to_codebook_scalar_quant(net_conv14, st, base, num_levels)

      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      if layer_quantized == 15:
        #codebook_conv_15 = tf.get_variable(name='conv15/wz/codebook', shape=[512,])
        net_conv15 = mapping_to_codebook_scalar_quant(net_conv15, st, base, num_levels)

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15

      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net

      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        if layer_quantized == 16:
          #codebook_conv_16 = tf.get_variable(name='conv16/wz/codebook', shape=[512,])
          net_conv16 = mapping_to_codebook_scalar_quant(net_conv16, st, base, num_levels)


        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_16_quant_act_1(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')
      codebook_conv_1 = tf.get_variable(name='conv1/wz/codebook', shape=[256,])
      net_conv1 = mapping_to_codebook(net_conv1, codebook_conv_1)

      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')

      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')

      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_16_quant_act_2(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')

      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')
      codebook_conv_2 = tf.get_variable(name='conv2/wz/codebook', shape=[256,])
      net_conv2 = mapping_to_codebook(net_conv2, codebook_conv_2)

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')

      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')

      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_16_quant_act_all_layers(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           layer_quantized=-1):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_conv1 = slim.conv2d(inputs , 64, [3, 3], scope='conv1/conv1_1')
      if layer_quantized == 1 or layer_quantized == -1:
        codebook_conv_1 = tf.get_variable(name='conv1/wz/codebook', shape=[256,])
        net_conv1 = mapping_to_codebook(net_conv1, codebook_conv_1)

      net_conv2 = slim.conv2d(net_conv1 , 64, [3, 3], scope='conv1/conv1_2')
      if layer_quantized == 2 or layer_quantized == -1:
        codebook_conv_2 = tf.get_variable(name='conv2/wz/codebook', shape=[256,])
        net_conv2 = mapping_to_codebook(net_conv2, codebook_conv_2)

      net_p_1 = slim.max_pool2d(net_conv2, [2, 2], scope='pool1')

      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net_conv3 = slim.conv2d(net_p_1, 128, [3, 3], scope='conv2/conv2_1')
      if layer_quantized == 3 or layer_quantized == -1:
        codebook_conv_3 = tf.get_variable(name='conv3/wz/codebook', shape=[256,])
        net_conv3 = mapping_to_codebook(net_conv3, codebook_conv_3)

      net_conv4 = slim.conv2d(net_conv3, 128, [3, 3], scope='conv2/conv2_2')
      if layer_quantized == 4 or layer_quantized == -1:
        codebook_conv_4 = tf.get_variable(name='conv4/wz/codebook', shape=[256,])
        net_conv4 = mapping_to_codebook(net_conv4, codebook_conv_4)

      net_p_2 = slim.max_pool2d(net_conv4, [2, 2], scope='pool2')

      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net_conv5 = slim.conv2d(net_p_2,  256, [3, 3], scope='conv3/conv3_1')
      if layer_quantized == 5 or layer_quantized == -1:
        codebook_conv_5 = tf.get_variable(name='conv5/wz/codebook', shape=[256,])
        net_conv5 = mapping_to_codebook(net_conv5, codebook_conv_5)
	  
      net_conv6 = slim.conv2d(net_conv5,  256, [3, 3], scope='conv3/conv3_2')
      if layer_quantized == 6 or layer_quantized == -1:
        codebook_conv_6 = tf.get_variable(name='conv6/wz/codebook', shape=[256,])
        net_conv6 = mapping_to_codebook(net_conv6, codebook_conv_6)

      net_conv7 = slim.conv2d(net_conv6,  256, [3, 3], scope='conv3/conv3_3')
      if layer_quantized == 7 or layer_quantized == -1:
        codebook_conv_7 = tf.get_variable(name='conv7/wz/codebook', shape=[256,])
        net_conv7 = mapping_to_codebook(net_conv7, codebook_conv_7)


      net_p_3 = slim.max_pool2d(net_conv7, [2, 2], scope='pool3')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net_conv8 = slim.conv2d(net_p_3, 512, [3, 3], scope='conv4/conv4_1')
      if layer_quantized == 8 or layer_quantized == -1:
        codebook_conv_8 = tf.get_variable(name='conv8/wz/codebook', shape=[256,])
        net_conv8 = mapping_to_codebook(net_conv8, codebook_conv_8)

      net_conv9 = slim.conv2d(net_conv8, 512, [3, 3], scope='conv4/conv4_2')
      if layer_quantized == 9 or layer_quantized == -1:
        codebook_conv_9 = tf.get_variable(name='conv9/wz/codebook', shape=[256,])
        net_conv9 = mapping_to_codebook(net_conv9, codebook_conv_9)

      net_conv10 = slim.conv2d(net_conv9, 512, [3, 3], scope='conv4/conv4_3')
      if layer_quantized == 10 or layer_quantized == -1:
        codebook_conv_10 = tf.get_variable(name='conv10/wz/codebook', shape=[256,])
        net_conv10 = mapping_to_codebook(net_conv10, codebook_conv_10)


      net_p_4 = slim.max_pool2d(net_conv10, [2, 2], scope='pool4')

      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net_conv11 = slim.conv2d(net_p_4, 512, [3, 3], scope='conv5/conv5_1')
      if layer_quantized == 11 or layer_quantized == -1:
        codebook_conv_11 = tf.get_variable(name='conv11/wz/codebook', shape=[256,])
        net_conv11 = mapping_to_codebook(net_conv11, codebook_conv_11)

      net_conv12 = slim.conv2d(net_conv11, 512, [3, 3], scope='conv5/conv5_2')
      if layer_quantized == 12 or layer_quantized == -1:
        codebook_conv_12 = tf.get_variable(name='conv12/wz/codebook', shape=[256,])
        net_conv12 = mapping_to_codebook(net_conv12, codebook_conv_12)

      net_conv13 = slim.conv2d(net_conv12, 512, [3, 3], scope='conv5/conv5_3')
      if layer_quantized == 13 or layer_quantized == -1:
        codebook_conv_13 = tf.get_variable(name='conv13/wz/codebook', shape=[256,])
        net_conv13 = mapping_to_codebook(net_conv13, codebook_conv_13)

      net_p_5 = slim.max_pool2d(net_conv13, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net_conv14 = slim.conv2d(net_p_5, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      if layer_quantized == 14 or layer_quantized == -1:
        codebook_conv_14 = tf.get_variable(name='conv14/wz/codebook', shape=[256,])
        net_conv14 = mapping_to_codebook(net_conv14, codebook_conv_14)

      net_conv14_drop = slim.dropout(net_conv14, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')

      net_conv15 = slim.conv2d(net_conv14_drop, 4096, [1, 1], scope='fc7')
      if layer_quantized == 15 or layer_quantized == -1:
        codebook_conv_15 = tf.get_variable(name='conv15/wz/codebook', shape=[256,])
        net_conv15 = mapping_to_codebook(net_conv15, codebook_conv_15)

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      net = net_conv15

      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net

      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net_conv16 = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')

        if layer_quantized == 16 or layer_quantized == -1:
          codebook_conv_16 = tf.get_variable(name='conv16/wz/codebook', shape=[256,])
          net_conv16 = mapping_to_codebook(net_conv16, codebook_conv_16)


        if spatial_squeeze:
          net_final = tf.squeeze(net_conv16, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net_final
      return net_final, end_points, [net_conv1, net_conv2, net_conv3, net_conv4, net_conv5, net_conv6, net_conv7, net_conv8, net_conv9, net_conv10, net_conv11, net_conv12, net_conv13, net_conv14, net_conv15, net_conv16]
vgg_16.default_image_size = 224

def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
