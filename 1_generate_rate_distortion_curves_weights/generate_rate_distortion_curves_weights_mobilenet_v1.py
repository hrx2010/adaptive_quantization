import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import slim
import sys 
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[4])

SLIM_PATH = '/home/wangzhe/Documents/exp/exp_2019_4_p2/4_25_4_output_error_mobilenet_v1_wei_test2/slim/'
IMAGENET_VAL_PATH = '/home/wangzhe/Documents/data/ImageNet2012/val/'
FINAL_OUTPUT_PATH = '/home/wangzhe/Documents/exp/mobilenet_v1_data/last_layer_output/'
sys.path.append(SLIM_PATH)

from nets.mobilenet_v1 import *
from nets.vgg import *
from preprocessing import inception_preprocessing 
from preprocessing import vgg_preprocessing 
from preprocessing import preprocessing_factory

from nets import mobilenet_v1

def write_data_to_file(file_name, data):
	fid = open(file_name, "w")
	fid.write(data)
	fid.close()

def map_matrix_to_codebook_faster(matrix, codebook):
    org_shape = np.shape(matrix)
    codebook = np.asarray(codebook)
    codebook = np.squeeze(codebook)
    temp_mat = np.reshape(matrix, [-1])
    temp_mat = np.squeeze(temp_mat)
    len_codebook = np.shape(codebook)[0]
    len_mat = np.shape(temp_mat)[0]
    temp_mat = np.expand_dims(temp_mat, axis=1)
    codebook = np.expand_dims(codebook, axis=0)
    m = np.repeat(temp_mat, len_codebook, axis=1)
    c = np.repeat(codebook, len_mat, axis=0)
    
    assert(np.shape(m) == np.shape(c))
    d = np.abs(m - c)
    select_id = np.argmin(d, axis=1)
    
    new_mat = [c[enum, item] for enum, item in enumerate(select_id)]
    return np.reshape(new_mat, org_shape)

def map_matrix_to_codebook(matrix, codebook):
    org_shape = np.shape(matrix)
    codebook = np.asarray(codebook)
    temp_mat = np.reshape(matrix, [-1])
    new_mat = np.zeros_like(temp_mat)
    for i in range(len(temp_mat)):
        curr = temp_mat[i]
        idx = np.argmin(np.abs(codebook - curr))
        new_mat[i] = codebook[idx]
    return np.reshape(new_mat, org_shape)

def set_variable_to_tensor(sess, tensor, value):
    return sess.run(tf.assign(tensor, value))

def load_codebook(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []
    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster

def load_random_testing_images(list_file):
    data = read_all_lines(list_file)
    testing_images = []
    for eachline in data[0:]:
        c = int(eachline)
        testing_images.append(c)

    return testing_images

def preprocessing_variables(all_variables):
	variable_weights = []
	variable_codebooks = []
	
	for i in range(len(all_variables)):
		if "codebook" in all_variables[i].name:
			variable_codebooks.append(all_variables[i])

	for i in range(len(all_variables)):
		if "codebook" not in all_variables[i].name:
			variable_weights.append(all_variables[i])

	return variable_weights, variable_codebooks

def convert_to_1D_array(values):
	values_1D = np.reshape(values, [-1])
	return values_1D

def read_all_lines(filename):
	with open(filename) as f:
		data = f.readlines()
	return data

def load_all_weights_name():
	data = read_all_lines('list_mobilenet_v1_weights.txt')
	all_weights_names = []

	for eachline in data[0:]:
		all_weights_names.append(eachline.strip('\n'))

	return all_weights_names

def load_codebook_wei(codebook_file):
    data = read_all_lines(codebook_file)
    cluster = []

    for eachline in data[0:]:
        c = float(eachline)
        cluster.append(c)

    return cluster[0], cluster[1]

model='mobilenet_v1_1.0_224'
batch_size=100
n_images=100
imges_path=IMAGENET_VAL_PATH
val_file='imagenet_2012_validation_synset_labels_new_index.txt'

data = read_all_lines(val_file)
ground_truth = [int(x) for x in data]
PIE_TRUTH = [x for x in ground_truth]
checkpoint_file = '/home/wangzhe/Documents/exp/exp_2019_4_p2/4_22_1_output_wei_mobile_net_v1/mobilenet_v1_1.0_224/%s.ckpt' % (model)
wei_codebook_path = '/home/wangzhe/Documents/exp/exp_2019_4_p2/4_24_1_quant_mobilenet_v1_wei/codebooks_mobilenet_v1_wei'

layer_id = int(sys.argv[1])
dead_zone_level = int(sys.argv[2])
quant_level = int(sys.argv[3])

all_weights_names = load_all_weights_name()
dir_wei_codebooks = wei_codebook_path + '/' + all_weights_names[layer_id] + '_' + str(dead_zone_level) + '_' + str(quant_level) + '.cb'
quant_lambda, quant_delta = load_codebook_wei(dir_wei_codebooks)
############################# main functions ###############################
#### laod all original feature vectors from files
#### build quantized vgg graph

with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(
        is_training=False, weight_decay=0.0)):
	input_string = tf.placeholder(tf.string)
	input_images = tf.read_file(input_string)
	input_images = tf.image.decode_jpeg(input_images, channels=3)

	image_preprocessing_fn = preprocessing_factory.get_preprocessing('mobilenet_v1', is_training=False)
	processed_images = image_preprocessing_fn(input_images, 224, 224)
	processed_images = tf.expand_dims(processed_images, 0)

	logits, _ = mobilenet_v1.mobilenet_v1(processed_images, is_training=False, depth_multiplier=1.0, num_classes=1001, id_quant_wei_layer_input=layer_id, wei_quant_lambda_input=quant_lambda,  wei_quant_delta_input=quant_delta, wei_quant_levels_input=quant_level)
#	probabilities = tf.nn.softmax(logits)

variables_to_restore = slim.get_variables_to_restore()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =  tf.Session(config=config)
sess.run(tf.global_variables_initializer())
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
init_fn(sess)

######################## load final output layer #########################
final_output_layer_original = [0] * n_images

for id_images in range(n_images):
	filename_layer_final = FINAL_OUTPUT_PATH + 'act_layer_final_image_' + str(id_images) + '.dat'
	fid1 = open(filename_layer_final , 'rb')
	final_output_layer_original[id_images] = np.fromfile(fid1 , np.float32)
	fid1.close()

######################## compute output error ############################
final_output_layer_quantized = [0] * n_images

for i in range(n_images):
	img_path = IMAGENET_VAL_PATH + 'ILSVRC2012_val_%08d.JPEG' % (i + 1)
	final_output_layer_quantized[i] = sess.run(logits , feed_dict={input_string:img_path})

output_error = 0.0
for i in range(n_images):
	output_error = output_error + np.mean((final_output_layer_original[i] - final_output_layer_quantized[i])**2)
output_error = output_error / n_images

print('output error %f' % output_error)
print('output error %d %d %d: %f' % (layer_id , dead_zone_level, quant_level, output_error))
output_error_saved_path = 'results/output_error_' + str(layer_id) + '_' + str(dead_zone_level) + '_' + str(quant_level)
file_results = open(output_error_saved_path , 'w')
file_results.write(str(output_error) + '\n')
file_results.close()
