#include "pbf_delete.h"
#include "../../util/pbf_cuda_util.h"
#include <device_launch_parameters.h>

namespace pbf {
namespace cuda {

namespace {
__global__ void checkDeleteParticleCUDA(
	uint32_t* hash,
	const dom_dim* position,
	dom_dim domain_min,
	dom_dim domain_max,
	uint32_t delete_cell_id,
	uint32_t num_particle
	)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto p = position[index];
	if (p.x < domain_min.x || domain_max.x < p.x ||
		p.y < domain_min.y || domain_max.y < p.y ||
		p.z < domain_min.z || domain_max.z < p.z) {
		hash[index] = delete_cell_id;
	}

}
}	// end of unnamed ns

void checkDeleteParticle(
	uint32_t* hash,
	const dom_dim* position,
	const std::pair<dom_dim, dom_dim>& domain,
	uint32_t delete_cell_id,
	uint32_t num_particle
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	if (num_block > 0)
		checkDeleteParticleCUDA << < num_block, num_thread >> >
		(hash, position, domain.first, domain.second, delete_cell_id, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

namespace {

__device__ int my_max(int x, int y) {
	return (x > y) ? x : y;
}

__global__ void calculateDeleteNumberCUDA(
	uint32_t* deleted_num,
	const uint32_t* hash,
	uint32_t delete_cell_id,
	uint32_t num_particle
	)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto h1 = hash[index];
	auto h0 = hash[my_max(index-1, 0)];

	if (h1 == delete_cell_id) {	// deleted
		if (h0 != h1) {	// the first particle of deleted particles
			deleted_num[0] = num_particle - index;
		}
	}

}
}	// end of unnamed ns

void calculateDeleteNumber(
	uint32_t* deleted_num,
	const uint32_t* hash,
	uint32_t delete_cell_id,
	uint32_t num_particle
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	if (num_block > 0)
		calculateDeleteNumberCUDA << < num_block, num_thread >> >(deleted_num, hash, delete_cell_id, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

}	// end of cuda ns
}	// end of pbf ns

