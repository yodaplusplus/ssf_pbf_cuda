#include "pbf_fill.h"
#include "../pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

namespace pbf {
namespace cuda {

template<typename T>
__global__ void fillCUDA(T* target, T value, uint32_t num)
{
	uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= num) return;

	target[gid] = value;
}

template<typename T>
void _fill(T* target, T value, uint32_t num)
{
	//thrust::fill(thrust::device_ptr<T>(target), thrust::device_ptr<T>(target + num), value);

	if (num > 0) {
		uint32_t num_thread, num_block;
		computeGridSize(num, 256, num_block, num_thread);
		fillCUDA << < num_block, num_thread >> >(target, value, num);
	}

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void fill(dom_dim* target, dom_dim value, uint32_t num)
{
	_fill(target, value, num);
}

void fill(scalar_t* target, scalar_t value, uint32_t num)
{
	_fill(target, value, num);
}

void fill(uint32_t* target, uint32_t value, uint32_t num)
{
	_fill(target, value, num);
}

}	// end of cuda ns
}	// end of pbf ns
