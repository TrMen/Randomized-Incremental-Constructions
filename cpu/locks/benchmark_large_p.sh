mkdir -p outputs/large_p
rm outputs/large_p/*

# Tests for very high thread counts, naive version discontinued here

for (( i=512; i<=4096; i+=128 )); do
sample_count=1000
echo "large_p: n = 2500 collab $i threads"
./nQueens 2 $i 2500 $sample_count >> outputs/large_p/collab_512-4096_2500.txt
echo "large_p: n = 3500 collab $i threads"
./nQueens 2 $i 3500 $sample_count >> outputs/large_p/collab_512-4096_3500.txt
done


~                 
