#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include "randomness.h"
#include <omp.h>
#include <utility>
#include <chrono>

//#define DEBUG 1
#define BENCHMARK 1


unsigned long long steps = 0;


class Solution
{
   public:
   int n;

   int  * queen;
   bool * dp;
   bool * dn;

   int i;

   Solution(const Solution & a)
   {
      n = a.n;
      queen = new int[n];
      dp    = new bool[2*n-1];
      dn    = new bool[2*n-1];

      for (int j = 0 ; j < n ; ++j)
         queen[j] = a.queen[j];

      for (int j = 0 ; j < 2*n-1 ; ++j)
         dp[j] = a.dp[j];

      for (int j = 0 ; j < 2*n-1 ; ++j)
         dn[j] = a.dn[j];

      i = a.i;
   }

   Solution(int nn)
   {
      n = nn;
      queen = new int[n];
      dp    = new bool[2*n-1];
      dn    = new bool[2*n-1];

      i = 0;

      for (int j = 0 ; j < n ; ++j)
         queen[j] = j;
   }

   ~Solution()
   {
      delete[] queen;
      delete[] dp;
      delete[] dn;
   }

   void reset()
   {
      i = 0;

      for (int j = 0 ; j < 2*n-1 ; ++j)
      {
         dp[j] = false;
         dn[j] = false;
      }
   }

   bool increment(RNG & rng)
   {
      int k = i;
      for (int j = i ; j < n ; ++j)
      {
         int jp = i + queen[j];
         int jn = i - queen[j] + n-1;

         if (!dp[jp] && !dn[jn])
            std::swap(queen[k++],queen[j]);
      }

      if (!(k-i))
         return false;

      int j = i + (rng.random() % (k-i));
      std::swap(queen[i] , queen[j]);

      int ip = i + queen[i];
      int in = i - queen[i] + n-1;

      dp[ip] = true;
      dn[in] = true;
      ++i;

      return true;
   }

#ifdef DEBUG
   bool is_correct(){
	   bool* temp_dp = new bool[2*n-1];
	   bool* temp_dn = new bool[2*n-1];
	   for(int j = 0; j < 2*n-1; ++j){
		   temp_dp[j] = false;
		   temp_dn[j] = false;
	   }

	   for(int j = 0; j < n; ++j){
                   int jp = j + queen[j];
                   int jn = j - queen[j] + n-1;

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






class Instance
{
   private:

      RNG rng;

   public:

      bool active;

      Solution * solution;

      omp_lock_t copy_lock;

      Instance(RNG prng , int n)
      {
	 omp_init_lock(&copy_lock);     
         rng = prng;
         solution = new Solution(n);
         active = false;
      }

         

      void reset()
      {
	 set_copy_lock();

         solution->reset();
         active = true;
      	
	 reset_copy_lock();
      }

      void reset_lock_free()
      {
         solution->reset();
	 active = true;
      }

      bool increment()
      {
	 set_copy_lock();

         #pragma omp atomic
         ++steps;
	 active &= solution->increment(rng);

	 reset_copy_lock();
       	 
	 return active;
      }

      bool increment_lock_free()
      {
         #pragma omp atomic
	 ++steps;
	 active &= solution->increment(rng);
	 
	 return active;
      }

      void set_copy_lock(){
	      omp_set_lock(&copy_lock);
      }

      void reset_copy_lock(){
	      omp_unset_lock(&copy_lock);
      }

      ~Instance()
      {
         delete solution;
	 omp_destroy_lock(&copy_lock);
      }

      int current_row(){
	      return solution->i;
      }
      
      //Returns true if the parameter instance was active, and copies from it if it was
      bool copy_from_instance_if_active(Instance * instance){
	      instance->set_copy_lock();

	      if(instance->active){
	      	delete solution;
         	solution = new Solution(*instance->solution);
	 	active = true;
		
		instance->reset_copy_lock();
	      	return true;
	      }

	      instance->reset_copy_lock();
	      return false;
      }
#ifdef DEBUG
      void check_correctness(int id){
	      if(solution->is_correct()){
		      printf("Instance number %d is correct\n", id);
	      }
      }
#endif
};






class Swarm
{
   private:

      RNG rng;
      Instance ** instance;
      int size;

      int n;

   public:

      Swarm(int swarm_size , int nn , int sample)
      {
         n = nn;
         size = swarm_size;

         instance = new Instance*[size];
         for (int i = 0 ; i < size ; ++i)
            instance[i] = new Instance(RNG(i , sample) , nn);
	
         rng = RNG(size , sample);
      }

      ~Swarm()
      {
         for (int i = 0 ; i < size ; ++i)
            delete instance[i];
         delete[] instance;
      }


      void solve_parallel()
      {
         omp_set_num_threads(size);
	 bool success = false;
         #pragma omp parallel shared(success)
         {
            int id = omp_get_thread_num();
            bool extended = true;
            instance[id]->reset();
            while(!success){
               while(instance[id]->current_row() < n)
               {
                  if (success){
		     break;
		  }
		  
		  extended = instance[id]->increment();
                  if (!extended)
                  {
                     instance[id]->reset();
                     break;
                  }
               }
	       if(extended){
		       success = true;
	       }
	    }
#ifdef DEBUG
	  instance[id]->check_correctness(id);
#endif
	  	    
	 }
      }





      void solve_collaborative()
      {   
         omp_set_num_threads(size);

         bool success = false;
         #pragma omp parallel shared(success)
         {
	    int* indices = new int[size];
	    for(int k = 0; k < size; ++k){
		    indices[k] = k;
	    }

            int id = omp_get_thread_num();
            bool fail = false;
            
            while(!success){
               instance[id]->reset();
               while (instance[id]->current_row() < n)
               {
		  if(success){
			  #ifdef DEBUG
			  printf("Exiting because success is true\n");
			  #endif
			  break;
		  }

                  fail = !instance[id]->increment();

                  if (fail)
                  {
		  	for(int x = size-1; x >= 0; --x){
				int y = rng.random() % (x+1);
				if(instance[id]->copy_from_instance_if_active(instance[indices[y]])){
					break;
				}
				else{
					std::swap(indices[x], indices[y]);
				}
			}
			//If no instance to copy from was found, go to reset of instance
			if(!instance[id]->active){
				break;
			}
                  }
               }
	       if(instance[id]->current_row() == n){
		  success = true;
#ifdef DEBUG
		  instance[id]->check_correctness(id);
#endif
	       }	  
            }
	    delete[] indices;
         }
      }
};







int main(int argc, char** argv)
{
#ifdef BENCHMARK
   auto start = std::chrono::high_resolution_clock::now();
#endif

   int version = atoi(argv[1]);
   int instances = atoi(argv[2]);
   int problem_size = atoi(argv[3]);
   int samples = atoi(argv[4]);


   double avg = 0;

   for (int i = 0 ; i < samples ; ++i)
   {
      steps = 0;

      Swarm swarm = Swarm(instances , problem_size , i);
#ifdef DEBUG
      printf("Starting solution\n");
#endif
      switch (version)
      {
         case 1:
            swarm.solve_parallel();
            break;
         case 2:
            swarm.solve_collaborative();
            break;
      }

      avg += (steps/instances);
//      printf("%llu\n" , steps);

   }

   avg /= samples;
   printf("%i" , (int) avg);

#ifdef BENCHMARK
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;
   double seconds = elapsed.count();
   printf("\t%f\n", seconds/samples);
#else
  printf("\n");   
#endif

 
}





