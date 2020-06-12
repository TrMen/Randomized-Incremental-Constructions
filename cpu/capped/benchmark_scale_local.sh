mkdir -p outputs/scale_local
rm outputs/scale_local/*

# Benchmarks for varying N, less samples and bigger steps on high values to make bigger tests possible

for (( i=1; i<=128; i*=2 )); do
sample_count=1000
echo "scale_local: n = 500 collab 8 threads"
./nQueens 2 8 $i 500 $sample_count >> outputs/scale_local/collab_8_1-128_500.txt
echo "scale_local: n = 500 collab 16 threads"
./nQueens 2 16 $i 500 $sample_count >> outputs/scale_local/collab_16_1-128_500.txt
echo "scale_local: n = 500 collab 32 threads"
./nQueens 2 32 $i 500 $sample_count >> outputs/scale_local/collab_32_1-128_500.txt
echo "scale_local: n = 500 collab 64 threads"
./nQueens 2 64 $i 500 $sample_count >> outputs/scale_local/collab_64_1-128_500.txt
echo "scale_local: n = 500 collab 128 threads"
./nQueens 2 128 $i 500 $sample_count >> outputs/scale_local/collab_128_1-128_500.txt
done
