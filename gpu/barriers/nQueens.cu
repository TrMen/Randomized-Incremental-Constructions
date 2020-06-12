#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <math.h>
#include <unistd.h>
#include <cstdlib>
#include <omp.h>
#include "randomness.cuh"
#include "nQueens.cuh"

//#define DEBUG 1

__device__ __managed__ unsigned long long steps = 0; 

template<int N>
class Solution
{
   public:

   unsigned short queens[N];
   bool dp[2*N-1];
   bool dn[2*N-1];

   int i;

   __host__ __device__ Solution(const Solution<N> & a)
   {
      for (int j = 0 ; j < N ; ++j)
         queens[j] = a.queens[j];

      for (int j = 0 ; j < 2*N-1 ; ++j){
         dp[j] = a.dp[j];
      	 dn[j] = a.dn[j];
   }
      i = a.i;
   }

   __device__ __host__ Solution(){
      for(int j = 0; j < 2*N-1; ++j){
	      dp[j] = false;
	      dn[j] = false;
      }

      for (unsigned short j = 0 ; j < N ; ++j){
         queens[j] = j;
      }

      i = 0;
   }

   __device__ void reset()
   {
      i = 0;

      for (int j = 0 ; j < 2*N-1 ; ++j)
      {
         dp[j] = false;
         dn[j] = false;
      }
   }

   __device__ bool increment(RNG & rng)
   {
      int k = i;
      for (int j = i ; j < N ; ++j)
      {
         int jp = i + queens[j];
         int jn = i - queens[j] + N-1;

         if (!dp[jp] && !dn[jn]){
		 unsigned short tmp = queens[j];
		 queens[j] = queens[k];
		 queens[k] = tmp;

		 ++k;
	 }
      }

      if (k == i){
         return false;
      }

      int j = i + (rng.random() % (k-i));
      
      unsigned short tmp = queens[i];
      queens[i] = queens[j];
      queens[j] = tmp;

      int ip = i + queens[i];
      int in = i - queens[i] + N-1;

      dp[ip] = true;
      dn[in] = true;
      ++i;

      return true;
   }
#ifdef DEBUG 
   __device__ bool is_correct(){
	   bool* temp_dp = new bool[2*N-1];
	   bool* temp_dn = new bool[2*N-1];
	   for(int j = 0; j < 2*N-1; ++j){
		   temp_dp[j] = false;
		   temp_dn[j] = false;
	   }

	   for(int j = 0; j < N; ++j){

		   int jp = j + queens[j];
		   int jn = j - queens[j] + N-1;

		   if(temp_dp[jp] || temp_dn[jn]){
			delete[] temp_dp;
	 	   	return false;
		   }
		   temp_dp[jp] = true;
		   temp_dn[jn] = true;
		   }
	   delete[] temp_dn;
	   return true;
   }
#endif
};

template<int N>
class Instance
{
private:
	RNG rng;

public:
      bool active;

      Solution<N> solution;

      __device__ Instance(int id, int sample)
      {
         rng = RNG(id, sample);
         solution = Solution<N>();
         active = false;
      }

      __device__ void copyFrom(Instance & valid)
      {
         solution = Solution<N>(valid.solution);
	 active = true;
      }

      __device__ void reset()
      {
         solution.reset();
         active = true;
      }

      __device__ bool increment()
      {
         atomicAdd(&steps, 1);

         return (active = active && solution.increment(rng));
      }
#ifdef DEBUG
      __device__ void print_check(int id){
	      if(solution.is_correct()){
		      printf("Instance number: %d is correct.\n", id);
	      }
      }
#endif
};


template<int N>
__global__ void solve_naive_kernel(Instance<N>* instances, int sample){
	 uint32_t id = threadIdx.x;

	 __shared__ bool success;
	 success = false;

	 instances[id] = Instance<N>(id, sample);
	 bool fail = false;

	 __syncthreads();
        
	 while(!success)
	 {
		 for(int i = 0; i < N; ++i){
			 if(fail = !instances[id].increment()){
				 instances[id].reset();
				 break;
			 }
		 }
		 if(!fail){
			 success = true;
#ifdef DEBUG
			 instances[id].print_check(id);
#endif
		 }
	 }
}

template<int N>
__global__ void solve_collaborative_kernel(Instance<N>* instances, int sample, RNG rng,int* indices){	
	 uint32_t id = threadIdx.x;
	 uint32_t size = blockDim.x;

         __shared__ bool failAll;
	 failAll = true;

	 instances[id] = Instance<N>(id, sample);
	 bool fail = false;

	 //Make indices of threads for randomization for copying instances later. Do it here to do as much work before barrier as possible
	 int offset = id*size;
	 for(int j = 0; j < size; ++j){
	 	indices[offset + j] = j;
	 }

	 __syncthreads();
	while(failAll){
	instances[id].reset();
	for(int i = 0; i < N; ++i){
		failAll = true;
		
		fail = !instances[id].increment();

		__syncthreads(); //I think this should be somehow avoidable. Maybe needs O(n) time though
		if(!fail){
			failAll = false;
		}
		__syncthreads();

		if(!failAll){
			if(fail){
				//Go through indices in random order
				for(int x = size - 1; x >= 0; --x){
					int y = rng.random() % (x+1);
					if(instances[indices[offset+y]].active){
						instances[id].copyFrom(instances[indices[offset+y]]);
						break;
					}
					else{
					//Swap the index to the back
					indices[offset + x] = indices[offset + x] ^ indices[offset + y];
					indices[offset + y] = indices[offset + y] ^ indices[offset + x];
					indices[offset + x] = indices[offset + x] ^ indices[offset + y];
					}
				}
			}
		}
		else{
			break;
		}
	    }
	}
	//Output correct solutions to check if algorithm work}
#ifdef DEBUG
	instances[id].print_check(id);
#endif
}


template<int N>
class Swarm
{
   private:

      RNG rng;

      Instance<N> * instances;
      
      int size;
      int n;
      int sample;

   public:

      Swarm(int swarm_size, int psample)
      {
	 size = swarm_size;
	 sample = psample;

	 cudaMalloc(&instances, size*sizeof(Instance<N>));
      	 rng = RNG(size, sample);
      }

      ~Swarm()
      {
         cudaFree(instances);
      }

      void solve_naive(){
#ifdef DEBUG
	printf("Solving naively with %d threads on %d Queens \n", size, N);
#endif
	solve_naive_kernel<N><<<1, size>>>(instances, sample);
	cudaDeviceSynchronize();
      }

      void solve_collaborative(){
#ifdef DEBUG
      	printf("Solving collaboratively with %d threads on %d Queens \n", size, N);
#endif
	int* indices;
	cudaMalloc(&indices, size*size*sizeof(int));
	
	solve_collaborative_kernel<N><<<1, size>>>(instances, sample, rng, indices);
	cudaDeviceSynchronize();
	cudaFree(indices);
      }
};

template<int problem_size>
int test(int version, int instances, int sample_count)
{
   double avg = 0;

   for (int sample = 0 ; sample < sample_count ; ++sample)
   {
      steps = 0;

      Swarm<problem_size> swarm{instances, sample};

      switch (version)
      {
         case 1:
            swarm.solve_naive();
            break;
         case 2:
            swarm.solve_collaborative();
            break;
      }
      
      avg += (steps/instances);
//      printf("%llu\n" , steps);

   }

   avg /= sample_count;
   return avg;
   //printf("%i\n" , (int) avg);
}
template int test<500>(int version, int insantex, int sample_count);
