mkdir -p outputs/scale_n
rm outputs/scale_n/*

# Benchmarks for varying N, less samples and bigger steps on high values to make bigger tests possible

for (( i=10; i<=200; i+=1 )); do
sample_count=500
echo "scale_problem: n = $i naive 32 threads"
./nQueens 1 32 $i $sample_count >> outputs/scale_n/naive_32_10-600.txt
echo "scale_problem: n = $i collab 32 threads"
./nQueens 2 32 $i $sample_count >> outputs/scale_n/collab_32_10-600.txt
echo "scale_problem: n = $i naive 64 threads"
./nQueens 1 64 $i $sample_count >> outputs/scale_n/naive_64_10-600.txt
echo "scale_problem: n = $i collab 64 threads"
./nQueens 2 64 $i $sample_count >> outputs/scale_n/collab_64_10-600.txt
echo "scale_problem: n = $i naive 128 threads"
./nQueens 1 128 $i $sample_count >> outputs/scale_n/naive_128_10-600.txt
echo "scale_problem: n = $i collab 128 threads"
./nQueens 2 128 $i $sample_count >> outputs/scale_n/collab_128_10-600.txt
done

# 32 thread variant discontinued after 300 queens

for (( i=204; i<=600; i+=2 )); do
sample_count=250
echo "scale_problem: n = $i naive 64 threads"
./nQueens 1 64 $i $sample_count >> outputs/scale_n/naive_64_10-600.txt
echo "scale_problem: n = $i collab 64 threads"
./nQueens 2 64 $i $sample_count >> outputs/scale_n/collab_64_10-600.txt
echo "scale_problem: n = $i naive 128 threads"
./nQueens 1 128 $i $sample_count >> outputs/scale_n/naive_128_10-600.txt
echo "scale_problem: n = $i collab 128 threads"
./nQueens 2 128 $i $sample_count >> outputs/scale_n/collab_128_10-600.txt
done

