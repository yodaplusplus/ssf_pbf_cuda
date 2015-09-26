#include "pbf_density.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../interaction/cuda/pbf_neighbor_search_device_util.cuh"
#include "../../util/pbf_cuda_util.h"
#include <device_launch_parameters.h>

namespace pbf {
namespace cuda {

namespace {
__device__ scalar_t calcDensityPair(
	const dom_dim& self_pos,
	const dom_dim& pair_pos,
	scalar_t smoothing_length,
	scalar_t particle_mass)
{
	auto pos_diff = self_pos - pair_pos;
	auto r = glm::length(pos_diff);
	auto direction = pos_diff / r;

	typedef kernel::cuda::PBFKERNEL kernel_t;
	auto k = pbf::kernel::cuda::weight<kernel_t>(r, smoothing_length);

	return k * particle_mass;
}

__device__ scalar_t sumDensity(
	//const std::vector<scalar_t>& kernels,
	const dom_dim* position,
	const uint32_t* neighbor_list,
	const dom_dim& self_pos,
	uint32_t self_index,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	uint32_t max_pair_particle_num)
{
	auto sum_density = 0.f;
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t pair_index = getNeighborParticleAddr(neighbor_list, self_index, pair_cnt, max_pair_particle_num);
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			sum_density += calcDensityPair(self_pos, pair_pos, smoothing_length, particle_mass);
			pair_cnt++;
		}
		else
			break;
	}

	return sum_density;
	//return (float)pair_cnt;
}

__global__ void calcDensityCUDA(
	const dom_dim* position,
	const uint32_t* neighbor_list,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	const auto self_pos = position[index];

	auto density = sumDensity(position, neighbor_list, self_pos, index, smoothing_length, particle_mass, max_pair_particle_num);

	//printf("density: %f\n", density);
}

}	// end of unnamed ns


void calcDensity(
	const dom_dim* position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	int num_particle
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

	if (num_block > 0)
		calcDensityCUDA<<< num_block, num_thread >>>
		(position, neighbor_list, smoothing_length, particle_mass, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns

