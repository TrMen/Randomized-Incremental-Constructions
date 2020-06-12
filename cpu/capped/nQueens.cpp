#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include "randomness.h"
#include <omp.h>
#include <utility>
#include <chrono>

//#define DEBUG 1
#define BENCHMARK 1
//#define MAXDEBUG 1

unsigned long long steps = 0;


class Solution
{
   public:
   int n;

   int  * queen;
   bool * dp;
   bool * dn;

   int i;

   bool active;

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
      active = true;
   }

   Solution(int nn)
   {
      n = nn;
      queen = new int[n];
      dp    = new bool[2*n-1];
      dn    = new bool[2*n-1];

      i = 0;
      active = true;

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
      active = true;
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
         return (active = false);

      int j = i + (rng.random() % (k-i));
      std::swap(queen[i] , queen[j]);

      int ip = i + queen[i];
      int in = i - queen[i] + n-1;

      dp[ip] = true;
      dn[in] = true;
      ++i;

      return (active = true);
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
      
      int* solution_indices;
      int* instance_indices;

      int instance_size;
      int swarm_size;

      int active_count;

      Solution ** solution;

      omp_lock_t copy_lock;

      RNG* rngs;

      Instance(RNG prng , int n, int pinstance_size, int pswarm_size, int id, int sample)
      {
	 omp_init_lock(&copy_lock);     
         rng = prng;
	 instance_size = pinstance_size;
	 swarm_size = pswarm_size;
	 active_count = instance_size;

	 rngs = new RNG[instance_size];
	 for(int j = 0; j < instance_size; ++j){
		 rngs[j] = RNG(id*instance_size+j, sample);
	 }

	 solution_indices = new int[instance_size];
	 for(int j = 0; j < instance_size; ++j){
		 solution_indices[j]=j;
	 }

	 instance_indices = new int[swarm_size];
	 for(int j = 0; j < swarm_size; ++j){
		 instance_indices[j]=j;
	 }

	 solution = new Solution*[instance_size];
	 for(int j = 0; j < instance_size; ++j){
         	solution[j] = new Solution(n);
	 }

      }

      void reset(int id)
      {
	 set_copy_lock();

	 #ifdef MAXDEBUG
	 printf("Resetting instance %d\n", id);
	 #endif
	
	 for(int sol_num = 0; sol_num < instance_size; ++sol_num){
         	solution[sol_num]->reset();
	 }
	 active_count = instance_size;
      	
	 reset_copy_lock();
      }

      bool increment(int sol_num, int id)
      {
	 set_copy_lock();

         #pragma omp atomic
         ++steps;
#ifdef MAXDEBUG
	 printf("Incrementing %d in instance %d. Current row: %d\n", sol_num, id, solution[sol_num]->i); 
#endif
	 bool extended = solution[sol_num]->increment(rngs[sol_num]);
	 if(!extended){
		--active_count;
	 }

	 reset_copy_lock();
       	 
	 return extended;
      }

      void set_copy_lock(){
	      omp_set_lock(&copy_lock);
      }

      void reset_copy_lock(){
	      omp_unset_lock(&copy_lock);
      }

      ~Instance()
      {
	 for(int j = 0; j < instance_size; ++j){
         	delete solution[j];
	 }
	 delete[] solution;
	
	 delete[] solution_indices;
	 delete[] instance_indices;

	 omp_destroy_lock(&copy_lock);
      }

      int current_row(int sol_num){
	      return solution[sol_num]->i;
      }
      
      //Returns true if the solution at source within this instance was active, and copies from it if it was
      bool copy_from_solution_if_active(int target, const Solution& source){
	      if(source.active){
	      	delete solution[target];
         	solution[target] = new Solution(source);
		++active_count;
	    	return true;
	      }

	      return false;
      }

#ifdef DEBUG
      void check_correctness(int id, int sol_num){
	      if(solution[sol_num]->is_correct()){
		      printf("Instance number %d is correct\n", id);
	      }
      }
#endif

      bool copy_from_within_instance(int target, int id){
#ifdef MAXDEBUG
	      printf("Starting copy from within instance %d, active_count: %d, target: %d\n", id, active_count, target);
	      printf("Active solutions:\t");
	      for(int j = 0; j < instance_size; ++j){
		      if(solution[j]->active){
			      printf("%d\t", j);
		      }
	      }
	      printf("\n");
#endif
	      set_copy_lock();
	      for(int x = instance_size-1; x >= 0; --x){
		      int source = rngs[target].random() % (x+1);
#ifdef MAXDEBUG
		      printf("attempting copy from within in instance %d from sol %d to sol %d\n", id, solution_indices[source], target);
#endif
		      if(copy_from_solution_if_active(target, *solution[solution_indices[source]])){
#ifdef MAXDEBUG
			 printf("Copied from within instance successfully, active_count: %d\n", active_count);
#endif
			 reset_copy_lock();
			 break;
		      }
		      else{
			std::swap(solution_indices[x], solution_indices[source]);
		      }
	      }
      }

      bool copy_from_instance(Instance* instance, int target_sol, int id){
#ifdef MAXDEBUG
	      printf("Entering copy_from_instance, own id: %d, own active count: %d instance->active_count: %d\n", id, active_count, instance->active_count);
	      printf("active solutions in outside instance:\t");
	      for(int j = 0; j < instance_size; ++j){
		      if(instance->solution[j]->active){
			      printf("%d\t", j);
		      }
	      }
	      printf("\n");
#endif
	     instance->set_copy_lock();
             if(instance->active_count > 0){
#ifdef MAXDEBUG
		printf("Entering copy selection loop\n");
#endif
      	     	for(int x = instance_size-1; x >= 0; --x){
			      int source = rngs[target_sol].random() % (x+1);
#ifdef MAXDEBUG
			      printf("We chose %d as solution to copy from\n", solution_indices[source]);
#endif
			      if(copy_from_solution_if_active(target_sol, *(instance->solution[solution_indices[source]]))){
#ifdef MAXDEBUG
			        printf("successfully copied into instance %d from outside instance from solution %d into solution %d\n", id, solution_indices[source], target_sol);
#endif
				instance->reset_copy_lock();
				return true;
			      }
			      else{
				std::swap(solution_indices[x], solution_indices[source]);
			      }
	      	}
	      }
	      instance->reset_copy_lock();
#ifdef MAXDEBUG
	      printf("Could not copy from an instance\n");
#endif
	      return false;
      }

      bool copy_from_outside_instance(Instance** instance, int target_sol, int id){
#ifdef MAXDEBUG
	      printf("Copying from outside instance into instance: %d\n", id);
#endif
	     for(int x = swarm_size-1; x >= 0; --x){
	     	int source = rngs[target_sol].random() % (x+1);
		if(copy_from_instance(instance[instance_indices[source]], target_sol, id)){
#ifdef MAXDEBUG
			printf("Copied from outside instance successfully\n");
#endif
			return true;
		}
		else{
			std::swap(instance_indices[x], instance_indices[source]);
		}
	     }
	     
	     return false;
      }

      bool copy_solutions_outside(Instance** instance, int id){
	#ifdef MAXDEBUG
	printf("Entering copy_solutions_outside\n");
	#endif
      	for(int target = 0; target < instance_size; ++target){
		if(!solution[target]->active && !copy_from_outside_instance(instance, target, id)){
			return false;
		}
	}
	return true;
      }

      void copy_solutions_within(int id){
      	for(int target = 0; target < instance_size; ++target){
		if(!solution[target]->active){
			copy_from_within_instance(target, id);
		}
	}
	#ifdef MAXDEBUG
	printf("Exiting copy_solutions_within\n");
	#endif
      }
};



class Swarm
{
   private:

      RNG rng;
      Instance ** instance;
      int size;

      int n;

   public:

      Swarm(int swarm_size , int nn , int sample, int instance_size)
      {
         n = nn;
         size = swarm_size;

         instance = new Instance*[size];
         for (int i = 0 ; i < size ; ++i)
            instance[i] = new Instance(RNG(i , sample) , nn, instance_size, size, i, sample);
	
         rng = RNG(size , sample);
      }

      ~Swarm()
      {
         for (int i = 0 ; i < size ; ++i)
            delete instance[i];
         delete[] instance;
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
            bool failAny = false;
	    
#ifdef MAXDEBUG
	    printf("Entering while loop\n");
#endif
	    while(!success){
		  for(int sol_num = 0; sol_num < instance[id]->instance_size; ++sol_num){
		  	failAny = false;
			if(success){
				#ifdef DEBUG
				printf("Terminating because of success true\n");
				#endif
			  	goto terminate;
		  	}

                  	if(!instance[id]->increment(sol_num, id)){
				failAny = true;
			}
		  
		  	if(instance[id]->current_row(sol_num) == n){
		  		success = true;
#ifdef DEBUG
		  		instance[id]->check_correctness(id, sol_num);
				printf("Terminating because row is n\n");
#endif
				goto terminate;
	         	}
		 }
		 if(failAny){
		 	if(instance[id]->active_count > 0){
				instance[id]->copy_solutions_within(id);
			 }
			 else if(!instance[id]->copy_solutions_outside(instance, id)){
					instance[id]->reset(id);
			 }
		 }
               }
terminate:
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
   int swarm_size = atoi(argv[2]);
   int instance_size = atoi(argv[3]);
   int problem_size = atoi(argv[4]);
   int samples = atoi(argv[5]);


   double avg = 0;

   for (int i = 0 ; i < samples ; ++i)
   {
      steps = 0;

      Swarm swarm = Swarm(swarm_size , problem_size , i, instance_size);
#ifdef DEBUG
      printf("Starting solution with %d threads and instance size of %d \n", swarm_size, instance_size);
#endif
      switch (version)
      {
         case 2:
            swarm.solve_collaborative();
            break;
      }

      avg += (steps/(swarm_size*instance_size));

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





