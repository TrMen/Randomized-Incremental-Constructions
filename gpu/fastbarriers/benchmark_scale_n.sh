mkdir -p outputs/scale_n
rm outputs/scale_n/*

# Benchmarks for varying N, less samples and bigger steps on high values to make bigger tests possible

for (( i=480; i<=500; i+=10 )); do
sample_count=250
echo "scale_problem: n = $i naive 256 threads"
./nQueens 1 1 256 $i $sample_count >> outputs/scale_n/naive_256_300-600.txt
echo "scale_problem: n = $i collab 256 threads"
./nQueens 2 1 256 $i $sample_count >> outputs/scale_n/collab_256_300-600.txt

echo "scale_problem: n = $i naive 512 threads"
./nQueens 1 1 512 $i $sample_count >> outputs/scale_n/naive_512_300-600.txt
echo "scale_problem: n = $i collab 512 threads"
./nQueens 2 1 512 $i $sample_count >> outputs/scale_n/collab_512_300-600.txt

echo "scale_problem: n = $i naive 1024 threads"
./nQueens 1 1 1024 $i $sample_count >> outputs/scale_n/naive_1024_300-600.txt
echo "scale_problem: n = $i collab 1024 threads"
./nQueens 2 1 1024 $i $sample_count >> outputs/scale_n/collab_1024_300-600.txt
done


