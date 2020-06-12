g++ -O3 -o nQueens nQueens.cpp -fopenmp -std=c++17

mkdir -p outputs
rm outputs/*

source benchmark_scale_local.sh

