id_gpu=0

for i in {0..6..1}
do 
	
		for k in 3 5 7 9 11 15 19 23 27 31 41 51 61 71 81 91 101 121 141 161 181 201 221 241 261 281 301 321 341 361 381 401 451 501 601 701 801 901 1001 2048 4096 8192 16384 32768 65536
		do
			python mobilenet_v1_original.py $i $k $id_gpu
		done
	
done