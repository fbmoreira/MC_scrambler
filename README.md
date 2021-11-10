# MC_scrambler
Reproduction of the ConGen v2 method described in "Efficient Generation of Application Specific Memory Controllers"

Basically, the idea of this repository is to replicate the algorithm described in the paper for any application.
The algorithm finds the min-k-set for a list of addresses, searching for the bit positions for "rows" in DRAM that will cause the minimum number of row conflicts.


1. Prerequisites:
	1. gcc (tested with gcc 7.4.3 and 11.0)
	2. OpenMP (libgomp from gcc)
	3. Architectural support for AVX512 (the code relies heavily on avx512 for performance reasons)


2. Building:
	Simply type "make". We used a default Makefile for this project.

3. Usage:
	`./congen2 File #lines`

	where "File" is a plain file with the format of "string LLU" in each line.
	The "string" is one of the 4: R, P, I, WB. These indicate "Reads", "Prefetches", "Instructions", and "Writebacks".
	In the end, you really only need to specify a WB, since any other operation is a "Read" in DRAM.
	"#lines" specifies the number of lines in the file.
	We use this parameter to optimize some memory allocations throughout the code, such as the buffer where we will place the file contents.
	We defined the parameter bits (row bits, row bits to XOR with bank bits, bank bits) as macros to enable gcc to perform some optimizations and avoid allocation problems for multiple valid solutions.
	Change these macros accordingly and recompile to solve a specific problem (read paper for an interpretation of what each of these mean). 

4. Caveats:
	The original paper assumes a constrained environment with a single channel and does not specify cache line size.
We, on the other hand, are using this technique to find bits in possibly multiple memory controllers and memory channels, with a 64~B cache line size.
	Thus, in the processing of the file, we used a macro named "BIT_IGNORE" to strip out (>> BIT_IGNORE) the addresses out of memory channel bits and cache line position bits.
	The assumption for doing so is that we always want cache lines to be contiguous, and we always want channel bits in the lower position to interleave contiguous accesses in all the memory channels (this 2nd assumption may not hold for long stride programs).
	We manage our own memory in this code (pool_allocator.h). The reasons are mostly performance: repeatedly calling malloc can be a major source of inefficiency. However, we may end up allocating too much memory for sufficiently large files and threads.
	A single thread should not have any issues, but as each thread processes a different bank and row xor bits mask, they need to generate their own representation of address distribution in the banks, which makes the memory requirement rather large. The main issue is that the problem is NP-complete, and thus exponential. To avoid recursion, we transformed the recursive algorithm described in the paper into an iterative form (BB_iterative). However, the stack needed to keep track of the data between iterations (would-be recursions) also grows large as we increase the number of k bits searched for the problem.
	
	Most importantly, one needs to consider how the results are presented.
	We print the bank bits mask, the rows to be XOR'ed mask, and the bits to be used as row bits (Fbits).
	Bank bits and xor row bits shown in the masks never conflict.
	However, before calculating the Fbits, we remove the bank bits and the row xor bits from the address to simplify the kernel of BB_iterative.
	Thus, when looking at the results, it may seem like there are conflicts, but one must take into account that the Fbits are *displaced* accordingly to the bank bits and row xor bits.

	For instance, if bank bits mask is 7 (0000 0111 in the last byte) and row xor is 168 (1010 1000 in the last byte), an Fbit of 63 (0011 1111) is actually 3920(1111 0101 0000), since the bits in bank bits and row xor were originally removed. #TODO make the transformation in the code before printing result.



