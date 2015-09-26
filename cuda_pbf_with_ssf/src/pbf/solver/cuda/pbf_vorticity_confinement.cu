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
;
template<typename kernel_t>
__device__ dom_dim calcVorticityEach(uint32_t helper_index, const dom_dim& self_pos, const dom_dim& self_vel,
	const dom_dim* helper_pos, const dom_dim* helper_vel, scalar_t h) {
	auto pair_pos = helper_pos[helper_index];
	auto pair_vel = helper_vel[helper_index];

	auto pos_diff = pair_pos - self_pos;
	auto r = glm::length(pos_diff);
	auto vel_diff = pair_vel - self_vel;

	dom_dim vorticity(0.f);
	if (r > 0.f) {
		const auto direction = pos_diff / r;
		vorticity = glm::cross(vel_diff, direction) *  pbf::kernel::cuda::weight_deriv<kernel_t>(r, h);
	}

	return vorticity;
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
			sum_prop += calcVorticityEach<kernel_t>(i, self_pos, self_vel, helper_pos, helper_vel, h);
		}
	}
	return sum_prop;
}

template<typename kernel_t>
__device__ dom_dim calcContributionVorticity(
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
__global__ void calcVorticityCUDA(
	dom_dim* vorticity,
	const dom_dim* position,
	const dom_dim* velocity,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width, dom_udim grid_size,
	scalar_t h, scalar_t m, scalar_t inv_rho0,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto self_pos = position[index];
	auto self_vel = velocity[index];

	// Contribution Calculation
	auto v = calcContributionVorticity<kernel_t>(self_pos, cell_start, cell_end, cell_width, grid_size, self_vel, position, velocity, h);

	//if (isnan(v.x) || isnan(v.y) || isnan(v.z)) {
	//	vorticity[index] = dom_dim(0.f);
	//}
	//else {
		vorticity[index] = m * inv_rho0 * v;
	//}

}

}	// end of unnamed ns

namespace {
;
template<typename kernel_t>
__device__ dom_dim calcLocationVecEach(uint32_t helper_index, const dom_dim& self_pos,
	const dom_dim* helper_pos, const dom_dim* helper_vor, scalar_t h) {
	auto pair_pos = helper_pos[helper_index];
	auto pair_vor = helper_vor[helper_index];

	auto pos_diff = pair_pos - self_pos;
	auto r = glm::length(pos_diff);
	auto w = glm::length(pair_vor);

	dom_dim loc(0.f);
	if (r > 0.f) {
		const auto direction = pos_diff / r;
		loc = w * direction *  pbf::kernel::cuda::weight_deriv<kernel_t>(r, h);
	}

	return loc;
}

template<typename kernel_t>
__device__
dom_dim calcContributionWithinCell(
uint32_t start_index,
uint32_t end_index,
const dom_dim& self_pos,
const dom_dim* helper_pos, const dom_dim* helper_vor,
scalar_t h
)
{
	auto sum_prop = dom_dim(0.f);
	if (start_index != 0xFFFFFFFF) {
		// iterate over perticles in this cell
		for (auto i = start_index; i < end_index; ++i) {
			sum_prop += calcLocationVecEach<kernel_t>(i, self_pos, helper_pos, helper_vor, h);
		}
	}
	return sum_prop;
}

template<typename kernel_t>
__device__ dom_dim calcContributionLocationVec(
	const dom_dim& self_pos,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width,
	const dom_udim& grid_size,
	const dom_dim* helper_pos, const dom_dim* helper_vor,
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
				sum_prop += calcContributionWithinCell<kernel_t>(start_index, end_index, self_pos, helper_pos, helper_vor, h);
			}
		}
	}

	return sum_prop;
}

template<typename kernel_t>
__global__ void addVorticityConfinementCUDA(
	dom_dim* velocity,
	const dom_dim* vorticity,
	const dom_dim* position,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width, dom_udim grid_size,
	scalar_t h,
	scalar_t particle_mass,
	scalar_t inv_stable_density,
	scalar_t adjust_parameter,
	scalar_t time_step,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto self_pos = position[index];
	auto self_vor = vorticity[index];

	// Contribution Calculation
	auto N = calcContributionLocationVec<kernel_t>(self_pos, cell_start, cell_end, cell_width, grid_size, position, vorticity, h);

	if (glm::length(N) > 0.f) {
		N = N * particle_mass * inv_stable_density;
		N = N / glm::length(N);
		velocity[index] = velocity[index] + adjust_parameter * glm::cross(N, self_vor) / inv_stable_density * time_step;
	}
}

}	// end of unnamed ns

namespace pbf {
namespace cuda {
;
void calcVorticity(
	dom_dim* vorticity,
	const dom_dim* sorted_position,
	const dom_dim* sorted_velocity,
	neighbor_search& ns,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	scalar_t inv_stable_density,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	const auto& cell_start = ns.getCellStart();
	const auto& cell_end = ns.getCellEnd();
	const auto& cell_width = ns.getCellWidth();
	auto& grid_size = ns.getGridSize();

	if (num_block > 0)
		calcVorticityCUDA<kernel_t> << < num_block, num_thread >> >(vorticity, sorted_position, sorted_velocity, cell_start, cell_end, cell_width, grid_size,
		smoothing_length, particle_mass, inv_stable_density, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void addVorticityConfinement(
	dom_dim* velocity,
	const dom_dim* vorticity,
	const dom_dim* position,
	neighbor_search& ns,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	scalar_t inv_stable_density,
	scalar_t adjust_parameter,
	scalar_t time_step,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	const auto& cell_start = ns.getCellStart();
	const auto& cell_end = ns.getCellEnd();
	const auto& cell_width = ns.getCellWidth();
	auto& grid_size = ns.getGridSize();

	if (num_block > 0)
		addVorticityConfinementCUDA<kernel_t> << < num_block, num_thread >> >(velocity, vorticity, position, cell_start, cell_end, cell_width, grid_size,
		smoothing_length, particle_mass, inv_stable_density, adjust_parameter, time_step, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
