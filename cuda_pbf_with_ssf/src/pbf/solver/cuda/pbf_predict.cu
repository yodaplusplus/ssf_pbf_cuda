#include "pbf_predict.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

namespace pbf {
namespace cuda {

namespace {

// updated = a + b * c
__global__ void multiplyAddCUDA(dom_dim* updated, const dom_dim* a, dom_dim b, scalar_t c, uint32_t num)
{
	uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= num) return;

	dom_dim src = a[gid];
	
	dom_dim dst;
	dst.x = fmaf(b.x, c, src.x);
	dst.y = fmaf(b.y, c, src.y);
	dst.z = fmaf(b.z, c, src.z);

	updated[gid] = dst;
}

// updated = a + b * c
__global__ void multiplyAddCUDA(dom_dim* updated, const dom_dim* a, const dom_dim* b, scalar_t c, uint32_t num)
{
	uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= num) return;

	dom_dim _a = a[gid];
	dom_dim _b = b[gid];

	dom_dim dst;
	dst.x = fmaf(_b.x, c, _a.x);
	dst.y = fmaf(_b.y, c, _a.y);
	dst.z = fmaf(_b.z, c, _a.z);

	updated[gid] = dst;
}

} // end of unnamed ns

void applyExternalForce(
	dom_dim* interim_velocity,
	const dom_dim* velocity,
	dom_dim ext_force,
	scalar_t time_step,
	int num_particle
	)
{
	if (num_particle > 0) {
		uint32_t num_thread, num_block;
		computeGridSize(num_particle, 128, num_block, num_thread);
		multiplyAddCUDA <<< num_block, num_thread >>>(interim_velocity, velocity, ext_force, time_step, num_particle);
	}

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void predictPosition(
	dom_dim* interim_position,
	const dom_dim* position,
	const dom_dim* velocity,
	scalar_t time_step,
	int num_particle)
{
	if (num_particle > 0) {
		uint32_t num_thread, num_block;
		computeGridSize(num_particle, 128, num_block, num_thread);
		multiplyAddCUDA << < num_block, num_thread >> >(interim_position, position, velocity, time_step, num_particle);
	}

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
