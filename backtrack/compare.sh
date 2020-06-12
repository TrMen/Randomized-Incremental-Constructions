g++ -O3 -o backtrack backtrack.cpp
g++ -O3 -o ric single_nQueens.cpp

rm backtrack_4-20.txt
rm ric_4-20.txt


samples=10000
for i in {4..35}; do
	echo "comparing n = $i"
	./backtrack $i $samples >> backtrack_4-20.txt
	./ric $i $samples >> ric_4-20.txt
done
