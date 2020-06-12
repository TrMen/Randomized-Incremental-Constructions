mkdir -p outputs/large_n
rm outputs/large_n/*

# Tests for high queen count and high threads, naive version discontinued here

for (( i=1500; i<=2500; i+=5 )); do
sample_count=1000
echo "large_n: n = $i collab 256 threads"
./nQueens 2 256 $i $sample_count >> outputs/large_n/collab_256_1500-2500.txt
echo "large_n: n = $i collab 512 threads"
./nQueens 2 512 $i $sample_count >> outputs/large_n/collab_512_1500-2500.txt
done

