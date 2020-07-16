# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope

from tensorpack.tfutils.varreplace import remap_variables

# this is a testing for github
inf = 100000000
cnt_wei_layer = 0
cnt_act_layer = 0
wei_quant_lambda = []
wei_quant_delta = []
wei_quant_levels = []
act_quant_delta = []
act_quant_levels = []


def weights_name_check(testing_name):
	if ("conv" in testing_name) and ("/W" in testing_name) and ("Momentum" not in testing_name) and ("shortcut" not in testing_name):	
		return 1
	if ("linear/W" in testing_name) and ("Momentum" not in testing_name):
		return 1
	return 0


def quant_wei_uni_dead_zone(x):
	if(weights_name_check(x.name) == 1):
		global cnt_wei_layer
		global wei_quant_lambda
		global wei_quant_delta
		global wei_quant_levels
		print('the id of weights %d' % (cnt_wei_layer))
		if cnt_wei_layer >= 50:
			cnt_wei_layer -= 50
		num_quant_levels = wei_quant_levels[cnt_wei_layer]
		print('finish the id of weights')
		q_lambda = wei_quant_lambda[cnt_wei_layer]
		q_delta = wei_quant_delta[cnt_wei_layer]

		q_k = int((num_quant_levels - 1) / 2)

		@tf.custom_gradient
		def _quant_wei_uni_dead_zone(x):
			tensor_zeros = tf.multiply(x , 0.0)
			quant_values_abs = tf.abs(x)
			quant_values_signs = tf.sign(x)

			quant_values_divided_floored = tf.floor(tf.divide(tf.subtract(quant_values_abs , q_lambda) , q_delta))
			quant_values_divided_floored_clipped = tf.clip_by_value(quant_values_divided_floored , 0 , tf.cast(q_k - 1, tf.float32))
			quant_values = tf.multiply(tf.add(tf.multiply(quant_values_divided_floored_clipped , q_delta) , q_lambda + q_delta/2.0 ) , quant_values_signs)

			condition = tf.less(tf.abs(x) , q_lambda)
			quant_results = tf.where(condition , tensor_zeros , quant_values)
			global cnt_wei_layer
			cnt_wei_layer += 1
			return quant_results, lambda dy: dy

#		print('number of weights layers %d' % (cnt_wei_layer))
		return _quant_wei_uni_dead_zone(x)

	else:
		return x


def quant_act_uni(x , base , num_quant_levels):
  if inf == num_quant_levels:
    return x

  @tf.custom_gradient
  def _quant_act_uni(x):
    x_divided = tf.divide(x , base)
    x_divided_rounded = tf.round(x_divided)
    x_divided_rounded_clipped = tf.clip_by_value(x_divided_rounded , clip_value_min = 0, clip_value_max = tf.cast(num_quant_levels - 1, tf.float32))
    x_quant_values = tf.multiply(x_divided_rounded_clipped , base)

    return x_quant_values, lambda dy: dy

  return _quant_act_uni(x)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    global cnt_act_layer    
    global act_quant_delta
    global act_quant_levels

    if cnt_act_layer >= 49:
        cnt_act_layer -= 49

    shortcut = l
    with remap_variables(quant_wei_uni_dead_zone):
        l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = quant_act_uni(l, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
    cnt_act_layer += 1

    with remap_variables(quant_wei_uni_dead_zone):
        l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = quant_act_uni(l, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
    cnt_act_layer += 1

    with remap_variables(quant_wei_uni_dead_zone):
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    out_relu = tf.nn.relu(out)
    out_relu = quant_act_uni(out_relu, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
    cnt_act_layer += 1

    return out_relu

def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnext_32x4d_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out * 2, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * 2, 3, strides=stride, activation=BNReLU, split=32)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that this pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        global cnt_relu_layer
        print('relu output %d: %s %s' % (cnt_relu_layer , l.name , l.shape))
        cnt_relu_layer += 1

        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits

def resnet_backbone(image, num_blocks, group_func, block_func, wei_quant_lambda_input, wei_quant_delta_input, wei_quant_levels_input, act_quant_delta_input, act_quant_levels_input):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that this pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        global cnt_act_layer
        global wei_quant_lambda
        global wei_quant_delta
        global wei_quant_levels
        global act_quant_delta
        global act_quant_levels

	if cnt_act_layer >= 49:
		cnt_act_layer -= 49

        wei_quant_lambda = wei_quant_lambda_input
        wei_quant_delta = wei_quant_delta_input
        wei_quant_levels = wei_quant_levels_input
        act_quant_delta = act_quant_delta_input
        act_quant_levels = act_quant_levels_input

        with remap_variables(quant_wei_uni_dead_zone):
            l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = quant_act_uni(l, act_quant_delta[cnt_act_layer], act_quant_levels[cnt_act_layer])
        cnt_act_layer += 1

        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)

        with remap_variables(quant_wei_uni_dead_zone):
            logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits
