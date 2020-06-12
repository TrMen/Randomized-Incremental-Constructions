mkdir -p outputs/scale_p
rm outputs/scale_p/*

# Benchmarks for varying thread count, less samples and bigger steps for low values to make uninteresting results faster

for (( i=4; i<=64; i+=4 )); do
sample_count=250
echo "scale_threads: n = 200 naive $i threads"
./nQueens 1 $i 200 $sample_count >> outputs/scale_p/naive_1-192_200.txt
echo "scale_threads: n = 200 collab $i threads"
./nQueens 2 $i 200 $sample_count >> outputs/scale_p/collab_1-192_200.txt
echo "scale_threads: n = 400 naive $i threads"
./nQueens 1 $i 400 $sample_count >> outputs/scale_p/naive_1-192_400.txt
echo "scale_threads: n = 400 collab $i threads"
./nQueens 2 $i 400 $sample_count >> outputs/scale_p/collab_1-192_400.txt
done

for (( i=66; i<=128; i+=2 )); do
sample_count=250
echo "scale_threads: n = 200 naive $i threads"
./nQueens 1 $i 200 $sample_count >> outputs/scale_p/naive_1-192_200.txt
echo "scale_threads: n = 200 collab $i threads"
./nQueens 2 $i 200 $sample_count >> outputs/scale_p/collab_1-192_200.txt
echo "scale_threads: n = 400 naive $i threads"
./nQueens 1 $i 400 $sample_count >> outputs/scale_p/naive_1-192_400.txt
echo "scale_threads: n = 400 collab $i threads"
./nQueens 2 $i 400 $sample_count >> outputs/scale_p/collab_1-192_400.txt
done

for i in {129..256}; do
sample_count=500
echo "scale_threads: n = 200 naive $i threads"
./nQueens 1 $i 200 $sample_count >> outputs/scale_p/naive_1-192_200.txt
echo "scale_threads: n = 200 collab $i threads"
./nQueens 2 $i 200 $sample_count >> outputs/scale_p/collab_1-192_200.txt
echo "scale_threads: n = 400 naive $i threads"
./nQueens 1 $i 400 $sample_count >> outputs/scale_p/naive_1-192_400.txt
echo "scale_threads: n = 400 collab $i threads"
./nQueens 2 $i 400 $sample_count >> outputs/scale_p/collab_1-192_400.txt
done

