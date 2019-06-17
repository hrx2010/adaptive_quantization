# source codes of adaptive quantization

step 1: generate the rate-distortion curves of weights
 - run -> python ./1_generate_rate_distortion_curves_weights/mobilenet_v1_original.py para_layer para_dz para_quant_level id_gpu
 - para_layer: the quantization layer
 - para_dz: the dead zone ratio
 - para_quant_level: the number of quantization levels
 - id_gpu: which GPU to run
 
step 2: generate the rate-distortion curves of activations
 - run -> python ./1_generate_rate_distortion_curves_activations/mobilenet_v1_original.py para_layer para_quant_level id_gpu
 - para_layer: the quantization layer
 - para_quant_level: the number of quantization levels
 - id_gpu: which GPU to run
 
step 3: solve optimal bit allocation under paredo contidion:
 - compile -> pareto_condition.cpp
 - run -> pareto_condition
 
step 4: perform inference of unequal bit allocation framework on imagenet:
  CUDA_VISIBLE_DEVICES=$id_gpu python ./slim/nets/mobilenet_v1_eval.py \
	--checkpoint_dir=/home/wangzhe/Documents/exp/exp_2019_4_p2/4_23_2_eval_mobilenet_v1_testing_2/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
	--dataset_dir=$dataset_dir \
	--bit_allocation=bit_allocations/bit_allocations_$i.txt \
	--dir_weight_codebooks=/home/wangzhe/Documents/exp/mobilenet_v1_data/codebooks_mobilenet_v1_wei/ \
	--dir_activation_codebooks=/home/wangzhe/Documents/exp/mobilenet_v1_data/codebooks_mobilenet_v1_activations_uni \
	|& tee logs_results_$i.txt
  
  
