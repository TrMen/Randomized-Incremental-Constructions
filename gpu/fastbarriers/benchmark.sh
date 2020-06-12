nvcc -O3 -o nQueens nQueensSimple.cu 
rm outputs/*

source benchmark_scale_n.sh

