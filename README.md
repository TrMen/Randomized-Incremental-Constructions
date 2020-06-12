# Randomized-Incremental-Constructions
Solving the N-Queens problem using Randomized Incremental Constructions(RIC) in massively parallel scale was the topic of my bachelor thesis. 
The main idea is to improve a solution to CSPs (using the N-Queens problem as an example)
that relies on a randomized strategy, called Randomized Incremental Constructions, by collaborative solution sharing across many threads.
Details can be found in the included thesis, or a short synopsis in the included presentation.

The source code is in the included folders. Trials include three variants of a CPU implementation using OpenMP, and a CUDA GPU implementation.
Implementation notes can also be found in the thesis.

To run benchmarks, simply execute benchmark.sh in the appropriate folder.

A Backtrack Search implementation is provided to show how badly it performs compared to even single-threaded RIC execution. Run a comparison with compare.sh in that folder.
