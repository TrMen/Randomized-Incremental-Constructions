g++ -O3 -o nQueens nQueens.cpp -fopenmp -std=c++17

mkdir -p outputs
rm outputs/*

source benchmark_scale_p.sh

source benchmark_scale_n.sh

source benchmark_large_p.sh

source benchmark_large_n.sh
