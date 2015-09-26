#include "pbf_kernel_memorization.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include "../../interaction/cuda/pbf_neighbor_search_device_util.cuh"
#include <device_launch_parameters.h>

extern __constant__ scalar_t h;

namespace pbf {
namespace cuda {

namespace {
template<typename kernel_t>
__device__ void memorizeKernelPair(
	scalar_t* kernels,
	dom_dim* grad_kernels,
	uint32_t pair_index,
	const dom_dim& self_pos,
	const dom_dim& pair_pos)
{
	auto pos_diff = self_pos - pair_pos;
	auto r = glm::length(pos_diff);
	auto direction = pos_diff / r;

	//auto k = pbf::kernel::cuda::weight<kernel_t>(r, smoothing_length);
	auto k = pbf::kernel::cuda::weight<kernel_t>(r, h);

	dom_dim kg(0.f);
	if (r > 0.f) {
		kg = pbf::kernel::cuda::weight_deriv<kernel_t>(r, h) * direction;
	}

	kernels[pair_index] = k;
	grad_kernels[pair_index] = kg;
}

template<typename kernel_t>
__global__ void memorizeKernelCUDA(
	scalar_t* kernels,
	dom_dim* grad_kernels,
	const dom_dim* position,
	const uint32_t* neighbor_list,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	const auto self_pos = position[index];
	// Contribution Calculation
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t neigbor_list_index = getNeighborListIndex(index, pair_cnt, max_pair_particle_num);
		uint32_t pair_index = neighbor_list[neigbor_list_index];
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			memorizeKernelPair<kernel_t>(kernels, grad_kernels, neigbor_list_index, self_pos, pair_pos);
			pair_cnt++;
		}
		else
			break;
	}
}

}	// end of unnamed ns


void memorizeKernelCalc(
	scalar_t* kernels,
	dom_dim* grad_kernels,
	std::shared_ptr<neighbor_search>& ns,
	const dom_dim* position,
	scalar_t smoothing_length,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 192, num_block, num_thread);

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

	if (num_block > 0)
		memorizeKernelCUDA<kernel_t><<< num_block, num_thread >>>
		(kernels, grad_kernels, position, neighbor_list, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
