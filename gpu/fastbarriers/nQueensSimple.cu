#include <stdio.h>
#include <cstdlib>
#include "randomness.cuh"

short WARP_SIZE = 32;

//#define STEPS 1
#define BENCHMARK 1
//#define DEBUG 1
//#define MINDEBUG 1
//#define MAXDEBUG 1
//#define DEBUGFOUR 1

#ifdef BENCHMARK
#include <chrono>
#endif

#ifdef STEPS
__device__ __managed__ unsigned long long steps = 0;
#endif

__device__ __managed__ bool success = false;
__constant__ int thread_count = 0;

#ifdef MINDEBUG
__inline__  __device__ bool is_correct(short* queens, short n, int id, int thread_count){
//#ifdef MAXDEBUG 
//           for(int x = 0; x < n; ++x){
//		   printf("queen[%d]=%d\n", x, queens[x*thread_count+id]);
//	   }
//#endif	   
	   bool* temp_dp = new bool[2*n-1];
           bool* temp_dn = new bool[2*n-1];
           for(short j = 0; j < 2*n-1; ++j){
                   temp_dp[j] = false;
                   temp_dn[j] = false;
           }

           for(short j = 0; j < n; ++j){

                   short jp = j + queens[j*thread_count+id];
                   short jn = j - queens[j*thread_count+id] + n-1;

                   if(temp_dp[jp] || temp_dn[jn]){
                        delete[] temp_dp;
			delete[] temp_dn;
                        return false;
                   }
                   temp_dp[jp] = true;
                   temp_dn[jn] = true;
                   }
	   delete[] temp_dp;
           delete[] temp_dn;
           return true;
   }
#endif

__device__ __inline__ void setup(short* queens, bool* dp, bool* dn, short n, short sample, RNG* rng){
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
        short thread_count = blockDim.x * gridDim.x;

        // Setup. Memory access should be coalesced this way
        for(short i = 0; i < n; ++i){
                queens[i*thread_count+id] = i;
        }

        for(short i = 0; i < 2*n-1; ++i){
                dn[i*thread_count+id] = false;
                dp[i*thread_count+id] = false;
        }
        rng[id] = RNG(id, sample);
}

__device__ __inline__ bool increment(short* queens, bool* dp, bool* dn, short n, short& i, short id, RNG* rng){
#ifdef STEPS		
		atomicAdd(&steps, 1);
#endif
	short k = i;

	for (short j = i ; j < n ; ++j){
      		short jp = i + queens[j*thread_count+id];
      		short jn = i - queens[j*thread_count+id] + n-1;

      		if (!dp[jp*thread_count+id] && !dn[jn*thread_count+id]){
               		short tmp = queens[j*thread_count+id];
               		queens[j*thread_count+id] = queens[k*thread_count+id];
               		queens[k*thread_count+id] = tmp;

			++k;
       		}
	}
		
	if (k == i){
#ifdef DEBUGFOUR
		printf("Exiting increment as failure\n");
		for(int j = 0; j < n; ++j){
			printf("queen[%d] = %d\n", j, queens[j*thread_count+id]);
		}
#endif
	   return false;
	}		

	short j = (short)(i + (rng[id].random() % (k-i)));
	
	short tmp = queens[i*thread_count+id];
	queens[i*thread_count+id] = queens[j*thread_count+id];
	queens[j*thread_count+id] = tmp;

	short ip = i + queens[i*thread_count+id];
	short in = i - queens[i*thread_count+id] + n-1;

	dp[ip*thread_count+id] = true;
	dn[in*thread_count+id] = true;
			
	++i;

#ifdef MAXDEBUG
	printf("i after increment: %d\n", i);
#endif
#ifdef DEBUGFOUR
	for(int j = 0; j < n; ++j){
		printf("queens[%d] = %d\n", j, queens[j*thread_count+id]);
	}
#endif

	return true;
}

__device__ __inline__ void reset(bool* dp, bool* dn, short n, short& i, int id){
	for(short j = 0; j < 2*n-1; ++j){
		dp[j*thread_count+id] = false;
		dn[j*thread_count+id] = false;
	}
	i = 0;
}

__global__ void solve_naive_kernel(short* queens, bool* dp, bool* dn, RNG* rng, short n, short sample){
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	setup(queens, dp, dn, n, sample, rng);
	
	short i = 0;

	while(true){
		if(success){
#ifdef DEBUG
			printf("Breaking out of id %d because success is true\n", id);
#endif
			break;
		}

		if(!increment(queens, dp, dn, n, i, id, rng)){
			reset(dp, dn, n, i, id);
		}

		if(i == n){
#ifdef MINDEBUG
			printf("Id %d is correct?%c\n", id, is_correct(queens, n, id, thread_count)?'y':'n');
#endif
			success = true;
		}
	}

}

__device__ __inline__ bool copy_from_solution_if_active(short* queens, bool* dp, bool* dn, bool* active, short n, short source, short target){
#ifdef MAXDEBUG
		printf("Entering copy with source: %d, target %d\n",source, target);
		for(int j = 0; j < thread_count; ++j){
			printf("active[%d] = %c\n", j, active[j]?'y':'n');
		}
		for(int j = 0; j < n; ++j){
			printf("source queens[%d] = %d\n", j, queens[j*thread_count+source]);
		}
		for(int j = 0; j < n; ++j){
			printf("target queens[%d] = %d\n", j, queens[j*thread_count+target]);
		}
#endif

	if(active[source]){
		//Copy. Do naively, can not be coalesced like reset because that would require continuous ordering which breaks the increment coalescing

		for(int j = 0; j < n; ++j){
			queens[j*thread_count+target] = queens[j*thread_count+source];
		}
		for(int j = 0; j < 2*n-1; ++j){
			dp[j*thread_count+target] = dp[j*thread_count+source];
			dn[j*thread_count+target] = dn[j*thread_count+source];
		}
		//i should already be implicitly equivalent because of barriers
		active[target] = true;
#ifdef MAXDEBUG
		printf("\nafter copy:\n");
		for(int j = 0; j < n; ++j){
			printf("target queens[%d] = %d\n", j, queens[j*thread_count+target]);
		}
#endif
		return true;
	}
	return false;
}

__global__ void solve_collaborative_kernel(short* queens, bool* dp, bool* dn, short* indices, RNG* rng, short n, short sample, bool* active, RNG global_rng){
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	
	setup(queens, dp, dn, n, sample, rng);
	
	__shared__ bool failAll;
	failAll = true;

	//Make indices of threads for randomization for copying instances later. Do it here to do as much work before barrier as possible
        for(short j = 0; j < thread_count; ++j){
               indices[j*thread_count+id] = j;
        }

#ifdef DEBUG
	printf("finished collab setup\n");
#endif
	short i = 0;
	__syncthreads();
	//--------------------------------------------------------

	while(failAll){
		reset(dp, dn, n, i, id);
		active[id] = true;
		
        	while(i < n){
#ifdef DEBUGFOUR
			printf("i at the start of for loop iteration: %d\n", i);
#endif
        	        failAll = true;

			active[id] = increment(queens, dp, dn, n, i, id, rng);

        	        __syncthreads(); //I think this should be somehow avoidable. Maybe needs O(n) time though
        	        if(active[id]){
        	                failAll = false;
        	        }
			__syncthreads();
	                if(!failAll){
        	                if(!active[id]){
                	                //Go through indices in random order
                        	        for(int x = thread_count - 1; x >= 0; --x){
                                	        int y = global_rng.random() % (x+1);
                                        	if(copy_from_solution_if_active(queens, dp, dn, active, n, indices[y*thread_count+id], id)){
							++i;
                                        	        break;
                                        	}
                                        	else{
                                        		//Swap the index to the back
                                        		short tmp = indices[x*thread_count+id];
                                        		indices[x*thread_count+id] = indices[y*thread_count+id];
                                        		indices[y*thread_count+id] = tmp;
						}
                                	}
                        	}
                	}
               	 	else{
                        	break;
                	}
            	}
	}
#ifdef MINDEBUG
	if(is_correct(queens, n, id, thread_count)){
		printf("Id %d is correct\n", id);
	}
#endif
#ifdef DEBUG
		printf("Id %d is correct?%c\n", id, is_correct(queens, n, id, thread_count)?'y':'n');
#endif
}

__host__ void solve_collab(short n, int block_count, int threads_per_block, short sample)
{
	
	short* queens;
	bool* dp;
	bool* dn;
	RNG* rng;
	bool* active;
	short* indices;


	unsigned int h_thread_count = block_count*threads_per_block;
	cudaMemcpyToSymbol(thread_count, &h_thread_count, sizeof(int));
	
	RNG global_rng{h_thread_count, (unsigned int)sample};

	//Set shared memory to be small, I think this can work on this GPU with compute capability 3.5
	//I only use shared memory for warp reset coordination
	cudaFuncSetCacheConfig(solve_naive_kernel, cudaFuncCachePreferL1);
	
	cudaMalloc(&active, sizeof(bool)*h_thread_count);
	cudaMalloc(&indices, sizeof(short)*h_thread_count*h_thread_count);
	cudaMalloc(&queens, sizeof(short)*h_thread_count*n);
	cudaMalloc(&dp, sizeof(bool)*h_thread_count*(2*n-1));
	cudaMalloc(&dn, sizeof(bool)*h_thread_count*(2*n-1));
	cudaMalloc(&rng, sizeof(RNG)*h_thread_count);

#ifdef MINDEBUG
	printf("Entering collaborative kernel\n");
#endif

	solve_collaborative_kernel<<<block_count, threads_per_block>>>(queens, dp, dn, indices, rng, n, sample, active, global_rng); 
	cudaDeviceSynchronize();

#ifdef MINDEBUG
	printf("Exiting collaborative kernel\n");
#endif
	success = false;
	cudaFree(active);
	cudaFree(indices);
	cudaFree(rng);
	cudaFree(queens);
	cudaFree(dp);
	cudaFree(dn);
}
__host__ void solve_ric(short n, int block_count, int threads_per_block, short sample)
{
	
	short* queens;
	bool* dp;
	bool* dn;
	RNG* rng;

	int h_thread_count = block_count*threads_per_block;
	cudaMemcpyToSymbol(thread_count, &h_thread_count, sizeof(int));

	//Set shared memory to be small, I think this can work on this GPU with compute capability 3.5
	//I only use shared memory for warp reset coordination
	cudaFuncSetCacheConfig(solve_naive_kernel, cudaFuncCachePreferL1);
	

	cudaMalloc(&queens, sizeof(short)*h_thread_count*n);
	cudaMalloc(&dp, sizeof(bool)*h_thread_count*(2*n-1));
	cudaMalloc(&dn, sizeof(bool)*h_thread_count*(2*n-1));
	cudaMalloc(&rng, sizeof(RNG)*h_thread_count);
#ifdef MINDEBUG
	printf("Entering naive kernel\n");
#endif
	solve_naive_kernel<<<block_count, threads_per_block>>>(queens, dp, dn, rng, n, sample); 
	cudaDeviceSynchronize();
#ifdef MINDEBUG
	printf("Exiting naive kernel\n");
#endif	
	success = false;
	cudaFree(rng);
	cudaFree(queens);
	cudaFree(dp);
	cudaFree(dn);
}

int main(int argc, char** argv){
#ifdef BENCHMARK
   auto start = std::chrono::high_resolution_clock::now();
#endif

   int version = atoi(argv[1]);
   int block_count = atoi(argv[2]);
   int threads_per_block = atoi(argv[3]);
   short problem_size = atoi(argv[4]);
   short samples = atoi(argv[5]);
   
   for(short sample = 0; sample < samples; ++sample){
	switch(version){
        	case 1: solve_ric(problem_size, block_count, threads_per_block, sample);
			break;
		case 2: solve_collab(problem_size, block_count, threads_per_block, sample);
			break;
	}
#ifdef STEPS
	steps /= threads_per_block*block_count;
#endif
   }
#ifdef STEPS
   printf("%llu\n", steps);
   printf("%llu", (steps/samples));
#endif

#ifdef BENCHMARK
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;
   double seconds = elapsed.count();
   printf("\t%f\n", seconds/samples);
#else
   printf("\n");
#endif

}
