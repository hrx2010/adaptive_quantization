
dataset_dir=/home/wangzhe/Documents/data/ImageNet2012/tfrecords/tfrecords
id_gpu=0

for i in {15..20..1}
do
	CUDA_VISIBLE_DEVICES=$id_gpu python ./slim/nets/mobilenet_v1_eval.py \
	--checkpoint_dir=/home/wangzhe/Documents/exp/exp_2019_4_p2/4_23_2_eval_mobilenet_v1_testing_2/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt \
	--dataset_dir=$dataset_dir \
	--bit_allocation=bit_allocations/bit_allocations_$i.txt \
	--dir_weight_codebooks=/home/wangzhe/Documents/exp/mobilenet_v1_data/codebooks_mobilenet_v1_wei/ \
	--dir_activation_codebooks=/home/wangzhe/Documents/exp/mobilenet_v1_data/codebooks_mobilenet_v1_activations_uni \
	|& tee logs_results_$i.txt
done
