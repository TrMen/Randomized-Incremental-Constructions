#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <stdlib.h>
#include <chrono>
#include <utility>

//#define minDE 1

unsigned long long steps = 0;

class Solution
{
   public:
   int n;

   int  * queen;
   bool * dp;
   bool * dn;

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
      for (int j = 0 ; j < 2*n-1 ; ++j)
      {
         dp[j] = false;
         dn[j] = false;
      }
   }

   bool increment(int i)
   {
      ++steps;

      int k = i;
      for (int j = i ; j < n ; ++j)
      {
         int jp = i + queen[j];
         int jn = i - queen[j] + n-1;

         if (!dp[jp] && !dn[jn]){
		 std::swap(queen[k++], queen[j]);
	 }
      }

      if (!(k-i))
         return false;

      int j = i + (rand() % (k-i));
      
      std::swap(queen[i] , queen[j]);

      int ip = i + queen[i];
      int in = i - queen[i] + n-1;

      dp[ip] = true;
      dn[in] = true;
      ++i;

      return true;
   }
#ifdef minDE
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

   
  void solve_ric(){
	bool fail = false;
	while(true){
		for(int i = 0; i < n; ++i){
			if(fail = !increment(i)){
				reset();
				break;
			}
		}
		if(!fail){
			break;
		}
  	}
#ifdef minDE
	printf("%c", is_correct()?'y':'n');
#endif
  }
};

int main(int argc, char** argv)
{
   int problem_size = atoi(argv[1]);
   int samples = atoi(argv[2]);

   auto start = std::chrono::high_resolution_clock::now();

   for (int i = 0 ; i < samples ; ++i)
   {
	srand(i);
	Solution s{problem_size};
	s.solve_ric();
   }

   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> elapsed = end - start;
   printf("%llu\n", steps/samples);
}





