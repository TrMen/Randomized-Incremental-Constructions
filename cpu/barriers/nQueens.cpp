#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <math.h>
#include <unistd.h>
#include <cstdlib>
#include <omp.h>
#include "randomness.h"


#define OUTPUT 1


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

};






class Instance
{
   private:

      RNG rng;

   public:

      bool active;

      Solution * solution;

      Instance(RNG prng , int n)
      {
         rng = prng;
         solution = new Solution(n);
         active = false;
      }

      void copyFrom(Instance * valid)
      {
         delete solution;
         solution = new Solution(*valid->solution);
	 active = true;
      }

      bool reset()
      {
         solution->reset();
         active = true;
      }

      bool increment()
      {
         #pragma omp atomic
         ++steps;

         return (active = active && solution->increment(rng));
      }

      ~Instance()
      {
         delete solution;
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
            bool fail = false;
            instance[id]->reset();
            while(true){
               for (int i = 0 ; i < n ; ++i)
               {
                  if (success){
		  	goto terminate;
		  }

                  if (fail = !instance[id]->increment())
                  {
                     instance[id]->reset();
                     break;
                  }
               }
               #pragma omp atomic
               success |= !fail;
	    }
	  terminate:
	  fail = !success;
	  	    
	 }
      }





      void solve_collaborative()
      {   
         omp_set_num_threads(size);

         bool failAll = true;
         #pragma omp parallel shared(failAll)
         {
	    int* indices = new int[size];
	    for(int k = 0; k < size; ++k){
		    indices[k] = k;
	    }

            int id = omp_get_thread_num();
            bool fail = false;
            
            while(failAll){
               instance[id]->reset();
               for (int i = 0 ; i < n ; ++i)
               {
                  fail = !instance[id]->increment();
                  #pragma omp single
                  failAll = true;


                  if(!fail){
                  	failAll = false;
		  }

                  #pragma omp barrier

                  if (!failAll)
                  {
                     if (fail)
                     {
			     for(int x = size-1; x > 0; --x){
				     int y = rng.random() % (x+1);
				     if(instance[indices[y]]->active){
					     instance[id]->copyFrom(instance[indices[y]]);
					     break;
				     }
				     else{
					     //Swap index to the end
					     std::swap(indices[x], indices[y]);
				     }
			     }
                     }
		  #pragma omp barrier
                  }
                  else
                     break;
               }     
            }
	    delete[] indices;
         }
      }
};







int main(int argc, char** argv)
{
   int version = atoi(argv[1]);
   int instances = atoi(argv[2]);
   int problem_size = atoi(argv[3]);
   int samples = atoi(argv[4]);


   double avg = 0;

   for (int i = 0 ; i < samples ; ++i)
   {
      steps = 0;

      Swarm swarm = Swarm(instances , problem_size , i);

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
   printf("%i\n" , (int) avg);
}





