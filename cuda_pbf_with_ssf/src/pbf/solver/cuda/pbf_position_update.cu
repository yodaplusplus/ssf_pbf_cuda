#include "pbf_position_update.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include "../../interaction/cuda/pbf_neighbor_search_device_util.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
//#include <sm_35_intrinsics.h>

using namespace std;

__constant__ scalar_t h;
__constant__ scalar_t m;
__constant__ scalar_t inv_rho0;
__constant__ scalar_t inv_k;

namespace {
	__host__ __device__ inline scalar_t length_opt(const dom_dim& a)
	{
		auto b = a.x * a.x;
		auto c = fmaf(a.y, a.y, b);
		auto d = fmaf(a.z, a.z, c);
		auto e = sqrtf(d);
		return e;
	}
}	// end of unnamed ns

namespace pbf {
namespace cuda {

namespace {
template<typename kernel_t>
__device__ dom_dim calcPositionUpdatePair(const dom_dim& self_pos, scalar_t self_scale,
	const dom_dim& pair_pos, scalar_t pair_scale)
{
	auto pos_diff = self_pos - pair_pos;
	auto r = length_opt(pos_diff);
	auto direction = pos_diff / r;

	auto kr = kernel::cuda::weight<kernel_t>(r, h);
	auto kr2 = kr * kr;
	auto kr4 = kr2 * kr2;
	auto tensile_correction = 10.f * kr4 * inv_k;
	auto self_clamped_scale = (self_scale < 0.f) ? -self_scale : 0.f;
	auto support_clamped_scale = (pair_scale < 0.f) ? -pair_scale : 0.f;
	auto wd = kernel::cuda::weight_deriv<kernel_t>(r, h);
	auto clamped_scale = self_clamped_scale + support_clamped_scale;

	auto scale_correction = clamped_scale * tensile_correction;

	dom_dim position_update(0.f);
	if (r > 0.f)
		//position_update = -(self_scale + support_scale) * pbf::kernel::cuda::weight_deriv<kernel_t>(r, h) * direction;
		position_update = -(self_scale + pair_scale + scale_correction) * wd * direction;

	//printf("%f, %f, %f\n", position_update.x, position_update.y, position_update.z);

	return position_update;
}

template<typename kernel_t>
__global__ void calcPositionUpdateCUDA(
	dom_dim* pos_update,
	const dom_dim* position,
	const scalar_t* scaling_factor,
	const uint32_t* neighbor_list,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	const auto self_pos = position[index];
	const auto self_scale = scaling_factor[index];
	// Contribution Calculation
	dom_dim sum_position_update(0.f);
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t pair_index = getNeighborParticleAddr(neighbor_list, index, pair_cnt, max_pair_particle_num);
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			const auto pair_scale = scaling_factor[pair_index];
			dom_dim position_update = calcPositionUpdatePair<kernel_t>(self_pos, self_scale, pair_pos, pair_scale);
			sum_position_update += position_update;
			pair_cnt++;
		}
		else
			break;
	}

	pos_update[index] = sum_position_update * m * inv_rho0;
}
}	// end of unnamed ns

void calcPositionUpdate(
	dom_dim* position_update,
	const dom_dim* position, const scalar_t* scaling_factor,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	if (num_block > 0)
		calcPositionUpdateCUDA<kernel_t><<< num_block, num_thread >>>
		(position_update, position, scaling_factor, neighbor_list, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

namespace {
template<typename kernel_t>
__device__ dom_dim calcPositionUpdatePair(scalar_t self_scale, scalar_t pair_scale,
	scalar_t kernel, const dom_dim& grad_kernel)
{
	auto kr2 = kernel * kernel;
	auto kr4 = kr2 * kr2;
	auto tensile_correction = 10.f * kr4 * inv_k;
	auto self_clamped_scale = (self_scale < 0.f) ? -self_scale : 0.f;
	auto support_clamped_scale = (pair_scale < 0.f) ? -pair_scale : 0.f;
	auto clamped_scale = self_clamped_scale + support_clamped_scale;

	auto scale_correction = clamped_scale * tensile_correction;

	dom_dim position_update(0.f);
	position_update = -(self_scale + pair_scale + scale_correction) * grad_kernel;

	//printf("%f, %f, %f\n", position_update.x, position_update.y, position_update.z);

	return position_update;
}

template<typename kernel_t>
__global__ void calcPositionUpdateCUDA(
	dom_dim* pos_update,
	const scalar_t* scaling_factor,
	const scalar_t* kernels, const dom_dim* grad_kernels,
	const uint32_t* neighbor_list,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	const auto self_scale = scaling_factor[index];
	// Contribution Calculation
	dom_dim sum_position_update(0.f);
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t neigbor_list_index = getNeighborListIndex(index, pair_cnt, max_pair_particle_num);
		uint32_t pair_index = neighbor_list[neigbor_list_index];
		if (pair_index != 0xFFFFFFFF) {
			const auto kernel = kernels[neigbor_list_index];
			const auto pair_scale = scaling_factor[pair_index];
			const auto grad_kernel = grad_kernels[neigbor_list_index];
			dom_dim position_update = calcPositionUpdatePair<kernel_t>(self_scale, pair_scale, kernel, grad_kernel);
			sum_position_update += position_update;
			pair_cnt++;
		}
		else
			break;
	}

	pos_update[index] = sum_position_update * m * inv_rho0;
}
}	// end of unnamed ns

void calcPositionUpdate(
	dom_dim* position_update,
	const scalar_t* scaling_factor,
	const scalar_t* kernels, const dom_dim* grad_kernels,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	if (num_block > 0)
		calcPositionUpdateCUDA<kernel_t><<< num_block, num_thread >>>
		(position_update, scaling_factor, kernels, grad_kernels, neighbor_list, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

namespace pu {
void setConstantMemory(
	scalar_t arg_h,
	scalar_t arg_m,
	scalar_t arg_inv_rho0,
	scalar_t arg_k
	)
{
	auto arg_inv_k = 1.f / arg_k;
	cudaMemcpyToSymbol(h, &arg_h, sizeof(scalar_t));
	cudaMemcpyToSymbol(m, &arg_m, sizeof(scalar_t));
	cudaMemcpyToSymbol(inv_rho0, &arg_inv_rho0, sizeof(scalar_t));
	cudaMemcpyToSymbol(inv_k, &arg_inv_k, sizeof(scalar_t));
}
} // end of pu ns

} // end of cuda ns
} // end of pbf ns
