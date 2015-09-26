#pragma once
#include <stdint.h>
#include <algorithm>

//Round a / b to nearest higher integer value
inline uint32_t iDivUp(uint32_t a, uint32_t b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
inline void computeGridSize(uint32_t n, uint32_t blockSize, uint32_t &numBlocks, uint32_t &numThreads)
{
	if (n == 0) {
		numThreads = 0;
		numBlocks = 0;
	}
	else {
		numThreads = std::min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
