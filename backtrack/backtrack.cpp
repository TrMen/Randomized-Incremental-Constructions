#include <utility>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <stdlib.h>

//#define minDE 1
//#define DEBUG 1
//#define DE 1

unsigned long long steps = 0;

class Solution{
private:
	unsigned short* queens;
	bool* dp;
	bool* dn;
	int n;

	unsigned long long individual_steps = 0;

	int next_extension(int row, int current_extension){
		for(int j = current_extension; j < n; ++j){
			int jp = row + queens[j];
			int jn = row - queens[j] + n-1;

			if(!dp[jp] && !dn[jn]){
				return j;
			}
		}
		return n;
	}

	bool backtrack_recursive(int row = 0){
		++steps;
		++individual_steps;
		//Stop outliers after a lot of steps, this takes about 30 seconds on agamemnon
		if(individual_steps > 20000000000){
			return true;
		}
		if(row == n){
			return true;
		}
		int current_extension = next_extension(row, row);
		
		while(current_extension < n){
			//Place queen in column
			std::swap(queens[row], queens[current_extension]);
			int ip = row + queens[row];
			int in = row - queens[row] + n-1;
			dp[ip] = true;
			dn[in] = true;

			//Extend next row
			if(backtrack_recursive(row+1)){
				return true;
			}
			else{
				//undo queen position
				//std::swap(queens[row], queens[current_extension]);
				dp[ip] = false;
				dn[in] = false;
			}
			current_extension = next_extension(row, current_extension+1);
		}
		//No extension found
#ifdef DE
		printf("no extension found for row %d \n", row);
#endif
		return false;	
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
                   int jp = j + queens[j];
                   int jn = j - queens[j] + n-1;

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
   void shuffle(){
	   for(int i = n-1; i > 0; --i){
		   int j = rand() % (i+1);
		   std::swap(queens[i], queens[j]);
	   }
   }
	
public:
	Solution(int nn)
	:n(nn){
		
		queens = new unsigned short[n];
		for(int i = 0; i < n; ++i){
			queens[i] = i;
		}
	
		dp = new bool[2*n-1];
		dn = new bool[2*n-1];
		for(int i = 0; i < 2*n-1; ++i){
			dp[i] = false;
			dn[i] = false;
		}

		shuffle();
	}

	~Solution(){
		delete[] queens;
		delete[] dp;
		delete[] dn;
	}

	void solve(){
		backtrack_recursive();
#ifdef DE
		for(int i = 0; i < n; ++i){
			printf("queens[%d] = %d\n", i, queens[i]);
		}
#endif
#ifdef minDE
		printf("correct? %c\n", is_correct()?'y':'n');
#endif
	}

};	

int main(int argc, char** argv){
   	int problem_size = atoi(argv[1]);
	int samples = atoi(argv[2]);
	auto start = std::chrono::high_resolution_clock::now();
	
	for(int i = 0; i < samples; ++i){
		srand(i);
		Solution s{problem_size};
		s.solve();
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	printf("%llu\n", steps/samples);

}
