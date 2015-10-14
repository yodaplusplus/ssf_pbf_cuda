#include "pbf_xsph.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <sm_35_intrinsics.h>

using namespace std;

namespace {

template<typename kernel_t>
__device__ dom_dim calcXSPHEach(uint32_t helper_index, const dom_dim& self_pos, const dom_dim& self_vel,
	const dom_dim* helper_pos, const dom_dim* helper_vel, scalar_t h) {
	auto pair_pos = helper_pos[helper_index];
	auto pair_vel = helper_vel[helper_index];

	auto pos_diff = pair_pos - self_pos;
	auto r = glm::length(pos_diff);
	auto vel_diff = pair_vel - self_vel;

	const auto inv_h = 1.f / h;
	auto kr = pbf::kernel::cuda::weight<kernel_t>(r, inv_h);

	return vel_diff * kr;
}

template<typename kernel_t>
__device__
dom_dim calcContributionWithinCell(
uint32_t start_index,
uint32_t end_index,
const dom_dim& self_pos, const dom_dim& self_vel,
const dom_dim* helper_pos, const dom_dim* helper_vel,
scalar_t h
)
{
	auto sum_prop = dom_dim(0.f);
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; ++i) {
			sum_prop += calcXSPHEach<kernel_t>(i, self_pos, self_vel, helper_pos, helper_vel, h);
		}
	}
	return sum_prop;
}

template<typename kernel_t>
__device__ dom_dim calcContribution(
	const dom_dim& self_pos,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width,
	const dom_udim& grid_size,
	const dom_dim& self_vel,
	const dom_dim* helper_pos, const dom_dim* helper_vel,
	scalar_t h
	)
{
	auto grid = pbf::cuda::calcGridPos(self_pos, cell_width);

	auto sum_prop = dom_dim(0.f);
#pragma unroll
	for (int z = -1; z <= 1; ++z) {
#pragma unroll
		for (int y = -1; y <= 1; ++y) {
#pragma unroll
			for (int x = -1; x <= 1; ++x) {
				dom_idim neighbor_grid(grid.x + x, grid.y + y, grid.z + z);
				auto neighbor_grid_hash = pbf::cuda::calcGridHash(neighbor_grid, grid_size);
				auto start_index = cell_start[neighbor_grid_hash];
				auto end_index = cell_end[neighbor_grid_hash];
				sum_prop += calcContributionWithinCell<kernel_t>(start_index, end_index, self_pos, self_vel, helper_pos, helper_vel, h);
			}
		}
	}

	return sum_prop;
}

template<typename kernel_t>
__global__ void calcXSPHCUDA(
	dom_dim* new_vel,
	const dom_dim* pos,
	const dom_dim* old_vel,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width, dom_udim grid_size,
	scalar_t h, scalar_t m, scalar_t inv_rho0,
	scalar_t xsph_param,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto self_pos = pos[index];
	auto self_vel = old_vel[index];

	// Contribution Calculation
	auto v = calcContribution<kernel_t>(self_pos, cell_start, cell_end, cell_width, grid_size, self_vel, pos, old_vel, h);

	if (isnan(v.x) || isnan(v.y) || isnan(v.z)) {
		new_vel[index] = self_vel;
	}
	else {
		new_vel[index] = self_vel + xsph_param * m * inv_rho0 * v;
	}

}

}	// end of unnamed ns

namespace pbf {
namespace cuda {
;
void applyXSPH(
	dom_dim* new_velocity,
	const dom_dim* sorted_position,
	const dom_dim* old_sorted_velocity,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t smoothing_length,
	scalar_t xsph_parameter,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	const auto& cell_start = ns->getCellStart();
	const auto& cell_end = ns->getCellEnd();
	const auto& cell_width = ns->getCellWidth();
	auto& grid_size = ns->getGridSize();

	if (num_block > 0)
		calcXSPHCUDA<kernel_t> << < num_block, num_thread >> >
		(new_velocity, sorted_position, old_sorted_velocity, cell_start, cell_end, cell_width, grid_size,
		smoothing_length, particle_mass, inv_stable_density, xsph_parameter, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
