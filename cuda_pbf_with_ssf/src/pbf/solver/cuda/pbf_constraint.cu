#include "pbf_constraint.h"
#include "../../interaction/cuda/pbf_contribution.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include "../../interaction/cuda/pbf_neighbor_search_device_util.cuh"
#include "boundary/pbf_plane_boundary.cuh"
#include "boundary/pbf_sphere_boundary.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

using namespace std;

extern __constant__ scalar_t h;
extern __constant__ scalar_t m;
extern __constant__ scalar_t inv_rho0;
extern __constant__ scalar_t inv_k;

namespace {

// used for scaling factor calculation
class scaling_t {
public:
	scalar_t ks;	// kernel sum
	dom_dim kgs; // kernel gradient sum
	scalar_t kg2s;	// kernel gradient norm sum
	__host__ __device__ scaling_t() : ks(0.f), kgs(0.f), kg2s(0.f) {}
	__host__ __device__ scaling_t(scalar_t v) : ks(v), kgs(v), kg2s(v) {}
	__host__ __device__ scaling_t& operator+=(const scaling_t& obj) {
		this->ks += obj.ks;
		this->kgs += obj.kgs;
		this->kg2s += obj.kg2s;
		return *this;
	}
};

__host__ __device__ inline scalar_t dot_opt(const dom_dim& a, const dom_dim& b)
{
	auto c = a.x * b.x;
	auto d = fmaf(a.y, b.y, c);
	auto e = fmaf(a.z, b.z, d);
	return e;
}

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
#pragma region scaling_factor
template<typename kernel_t>
__device__ scaling_t calcScalingFactorPair(
	const dom_dim& self_pos,
	const dom_dim& pair_pos)
{
	auto pos_diff = self_pos - pair_pos;
	auto r = glm::length(pos_diff);
	auto direction = pos_diff / r;
	const auto inv_h = 1.f / h;

	//auto k = pbf::kernel::cuda::weight<kernel_t>(r, smoothing_length);
	auto k = pbf::kernel::cuda::weight<kernel_t>(r, inv_h);

	dom_dim kg(0.f);
	if (r > 0.f) {
		kg = pbf::kernel::cuda::weight_deriv<kernel_t>(r, inv_h) * direction;
	}

	auto kg2 = dot_opt(kg, kg);

	scaling_t v;
	v.ks = k;
	v.kgs = kg;
	v.kg2s = kg2;

	return v;
}

template<typename kernel_t>
__global__ void calcScalingFactorCUDA(
	scalar_t* scaling_factor,
	const dom_dim* position,
	const uint32_t* neighbor_list,
	scalar_t relaxation,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	const auto self_pos = position[index];
	// Contribution Calculation
	scaling_t sum_scaling_factor(0.f);
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t pair_index = getNeighborParticleAddr(neighbor_list, index, pair_cnt, max_pair_particle_num);
		if (pair_index != 0xFFFFFFFF) {
			const auto pair_pos = position[pair_index];
			scaling_t scaling_factor = calcScalingFactorPair<kernel_t>(self_pos, pair_pos);
			sum_scaling_factor += scaling_factor;
			pair_cnt++;
		}
		else
			break;
	}

	auto constraint = sum_scaling_factor.ks * m * inv_rho0 - 1.f;
	auto kg2s = sum_scaling_factor.kg2s + glm::dot(sum_scaling_factor.kgs, sum_scaling_factor.kgs);

	auto s = constraint / (m * m * pow(inv_rho0, 2) * kg2s + relaxation);

	scaling_factor[index] = s;
}
}	// end of unnamed ns

void calcScalingFactor(
	scalar_t* scaling_factor,
	const dom_dim* position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_particle
	)
{
	typedef kernel::cuda::PBFKERNEL kernel_t;

	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	using namespace std;
	auto neighbor_list = ns->getNeighborList();
	const auto max_pair_particle_num = ns->getMaxPairParticleNum();

	if (num_block > 0)
		calcScalingFactorCUDA<kernel_t><<< num_block, num_thread >>>
		(scaling_factor, position, neighbor_list, relaxation, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}


namespace {
template<typename kernel_t>
__device__ scalar_t calcScalingFactorPair(
	const dom_dim& grad_kernel)
{
	auto kg2 = fmaf(grad_kernel.z, grad_kernel.z, fmaf(grad_kernel.y, grad_kernel.y, grad_kernel.x * grad_kernel.x));
	//auto kg2 = dot_opt(grad_kernel, grad_kernel);
	return kg2;
}

template<typename kernel_t>
__global__ void calcScalingFactorCUDA(
	scalar_t* scaling_factor,
	const scalar_t* kernels,
	const dom_dim* grad_kernels,
	const uint32_t* neighbor_list,
	scalar_t relaxation,
	uint32_t max_pair_particle_num,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	// Contribution Calculation
	scaling_t sum_scaling_factor(0.f);
	uint32_t pair_cnt = 0;
	while (true) {
		uint32_t neigbor_list_index = getNeighborListIndex(index, pair_cnt, max_pair_particle_num);
		uint32_t pair_index = neighbor_list[neigbor_list_index];
		if (pair_index != 0xFFFFFFFF) {
			const auto grad_kernel = grad_kernels[neigbor_list_index];
			const auto kernel = kernels[neigbor_list_index];
			scalar_t kg2 = calcScalingFactorPair<kernel_t>(grad_kernel);
			sum_scaling_factor.ks += kernel;
			sum_scaling_factor.kgs += grad_kernel;
			sum_scaling_factor.kg2s += kg2;
			pair_cnt++;
		}
		else
			break;
	}

	auto constraint = fmaf(sum_scaling_factor.ks * m, inv_rho0, - 1.f);
	auto kg2s = fmaf(sum_scaling_factor.kgs.x, sum_scaling_factor.kgs.x, fmaf(sum_scaling_factor.kgs.z, sum_scaling_factor.kgs.z,
		fmaf(sum_scaling_factor.kgs.y, sum_scaling_factor.kgs.y, sum_scaling_factor.kg2s)));

	auto m2_rho2 = m * m * inv_rho0 * inv_rho0;
	auto s = constraint / (fmaf(m2_rho2, kg2s, relaxation));

	scaling_factor[index] = s;
}

}	// end of unnamed ns

void calcScalingFactor(
	scalar_t* scaling_factor,
	const scalar_t* kernels,
	const dom_dim* grad_kernels,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	scalar_t relaxation,
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
		calcScalingFactorCUDA<kernel_t><<< num_block, num_thread >>>
		(scaling_factor, kernels, grad_kernels, neighbor_list, relaxation, max_pair_particle_num, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

#pragma endregion

namespace {
	// intersection of p1-p2 line and plane, no exception handling
	__host__ __device__ dom_dim intersection(dom_dim p1, dom_dim p2, const glm::vec4& plane) {
		auto e = (p2 - p1) / glm::length(p2 - p1);
		auto abc = glm::vec3(plane.x, plane.y, plane.z);
		auto k = -(glm::dot(abc, p1) + plane.w) / glm::dot(abc, e);
		//auto qc = p1 + k * e;
		dom_dim qc;
		qc.x = fmaf(k, e.x, p1.x);
		qc.y = fmaf(k, e.y, p1.y);
		qc.z = fmaf(k, e.z, p1.z);

		return qc;
	}
} // end of unnamed ns

__global__ void responseCollisionCUDA(
	dom_dim* position_update,
	const dom_dim* predicted_position,
	const dom_dim* old_position,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto delta_p = position_update[index];
	auto predicted_x = predicted_position[index];
	auto old_x = predicted_position[index];
	//if (index == 0)
	//	printf("%f, %f, %f\n", delta_p.x, delta_p.y, delta_p.z);

	while (true) {
		auto p = predicted_x + delta_p;
		dom_dim sim_origin(2.2f);
		dom_dim sim_end(6.5f);
		bool collision_check = false;
#if 0
		// bottom
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_origin, dom_dim(0.f, 1.f, 0.f));
		// top
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_end, dom_dim(0.f, -1.f, 0.f));
		// left wall
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_origin, dom_dim(1.f, 0.f, 0.f));
		// right wall
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_end, dom_dim(-1.f, 0.f, 0.f));
		// front
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_origin, dom_dim(0.f, 0.f, 1.f));
		// back
		responsePlaneBoundary(collision_check, delta_p, old_x, p, sim_end, dom_dim(0.f, 0.f, -1.f));
#endif

#if 1
		// in a sphere
		responseInnerSphereBoundary(collision_check, delta_p, old_x, p, dom_dim(3.f, 3.f, 3.f), 3.f);

		p = predicted_x + delta_p;

		// sphere obstacles
		responseOuterSphereBoundary(collision_check, delta_p, old_x, p, dom_dim(3.f, 0.f, 3.f), 1.f);
#endif

		if (!collision_check) {
			break;
		}
	}
	position_update[index] = delta_p;
	//if (index == 0)
	//	printf("%f, %f, %f\n", delta_p.x, delta_p.y, delta_p.z);
}

void responseCollision(
	dom_dim* position_update,
	const dom_dim* predicted_position,
	const dom_dim* old_position,
	int num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	if (num_block > 0)
		responseCollisionCUDA << < num_block, num_thread >> >(position_update, predicted_position, old_position, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

__global__ void updateInterimPositionCUDA(
	dom_dim* position,
	const dom_dim* position_update,
	int num_particle
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	position[index] += position_update[index];
}

void updateInterimPosition(
	dom_dim* position,
	const dom_dim* position_update,
	int num_particle)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	if (num_block > 0)
		updateInterimPositionCUDA<< < num_block, num_thread >> >(position, position_update, num_particle);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns
