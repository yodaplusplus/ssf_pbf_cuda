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
	const dom_dim& pair_pos,
	scalar_t inv_h,
	scalar_t alpha_kernel,
	scalar_t alpha_kernel_deriv)
{
	auto pos_diff = self_pos - pair_pos;
	auto r2 = fmaf(pos_diff.z, pos_diff.z, fmaf(pos_diff.y, pos_diff.y, pos_diff.x * pos_diff.x));
	const auto inv_r = rsqrtf(r2);
	auto q2 = r2 * inv_h * inv_h;

	float k = 0.f;
	dom_dim kg(0.f);
	float kd = 0.f;
	
	// kernel
	if (q2 <= 1.f) {
		auto t = 1.f - q2;
		k = alpha_kernel * t * t * t;
	}

	// kernel derivative
	auto r = r2 * inv_r;
	auto u = fmaf(-r, inv_h, 1.f);
	if (inv_h <= inv_r) {
		kd = alpha_kernel_deriv * u * u;
	}

	//auto k = pbf::kernel::cuda::weight<kernel_t>(r, inv_h);
	if (r2 > 0.f) {
		auto direction = pos_diff * inv_r;
		kg = kd * direction;
		//kg = pbf::kernel::cuda::weight_deriv<kernel_t>(r, inv_h) * direction;
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

	const auto inv_h = 1.f / h;
	const auto self_pos = position[index];

	const auto inv_h2 = inv_h * inv_h;
	const auto alpha_kernel = 1.56668147065f * inv_h * inv_h2;
	const auto alpha_kernel_deriv = -14.3239448745f * inv_h2 * inv_h2;

	// Contribution Calculation
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t neigbor_list_index = getNeighborListIndex(index, pair_cnt, max_pair_particle_num);
		uint32_t pair_index = neighbor_list[neigbor_list_index];
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			memorizeKernelPair<kernel_t>(kernels, grad_kernels, neigbor_list_index, self_pos, pair_pos, inv_h, alpha_kernel, alpha_kernel_deriv);
			pair_cnt++;
		}
		else
			break;
	}
}

namespace opt {
// two thread per one particle, 128 thread per block
template<typename kernel_t>
__global__ void memorizeKernelCUDA(
	scalar_t* kernels,
	dom_dim* grad_kernels,
	const dom_dim* __restrict__ position,
	const uint32_t* __restrict__ neighbor_list,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	const auto block_index = index >> 7;	// 7 = log2(128)
	const auto local_index = index & 127;	// 0 ~ 127
	const auto input_index = (local_index & 63) + (block_index << 6);
	const auto group = local_index >> 6;

	if (input_index >= num_particle) return;

	const auto self_pos = position[input_index];
	const auto inv_h = 1.f / h;
	const auto inv_h2 = inv_h * inv_h;
	const auto alpha_kernel = 1.56668147065f * inv_h * inv_h2;
	const auto alpha_kernel_deriv = -14.3239448745f * inv_h2 * inv_h2;

	// Contribution Calculation
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t neigbor_list_index = getNeighborListIndex(input_index, 2 * pair_cnt + group, max_pair_particle_num);
		uint32_t pair_index = neighbor_list[neigbor_list_index];
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			memorizeKernelPair<kernel_t>(kernels, grad_kernels, neigbor_list_index, self_pos, pair_pos, inv_h, alpha_kernel, alpha_kernel_deriv);
			pair_cnt++;
		}
		else
			break;
	}
}
}	// end of opt ns

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
	computeGridSize(num_particle * 2, 128, num_block, num_thread);	// opt
	//computeGridSize(num_particle, 128, num_block, num_thread);	// no opt

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

	if (num_block > 0)
		opt::memorizeKernelCUDA<kernel_t><<< num_block, num_thread >>>
		(kernels, grad_kernels, position, neighbor_list, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
