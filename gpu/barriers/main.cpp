#include "nQueens.cuh"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv){
  int version = atoi(argv[1]);
  int instances = atoi(argv[2]);
  int sample_count = atoi(argv[3]);

  constexpr int problem_size = 500;

  printf("%d",test<problem_size>(version, instances, sample_count));

}
