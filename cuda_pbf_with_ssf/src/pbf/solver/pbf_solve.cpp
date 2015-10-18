#include "pbf_solve.h"
#include "cuda/pbf_predict.h"
#include "cuda/pbf_constraint.h"
#include "cuda/pbf_update.h"
#include "cuda/pbf_position_update.h"
#include "cuda/pbf_density.h"
#include "cuda/pbf_kernel_memorization.h"
#include "cuda/pbf_response_collision.h"
#include "../interaction/pbf_neighbor_search.h"
#include "../kernel/cuda/pbf_kernel.h"

using namespace std;

namespace pbf {

void predict(
	dom_dim* interim_position, dom_dim* interim_velocity,
	const dom_dim* position, const dom_dim* velocity,
	dom_dim ext_force, scalar_t time_step,
	int num_particle
	)
{
	cuda::applyExternalForce(interim_velocity, velocity, ext_force, time_step, num_particle);
	cuda::predictPosition(interim_position, position, interim_velocity, time_step, num_particle);
}

void solveConstraint(
	dom_dim* interim_position, dom_dim* position_update, scalar_t* scaling_factor,
	const dom_dim* old_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_iteration,
	int num_particle
	)
{
	auto& h = smoothing_length;
	const auto inv_h = 1.f / h;
	static auto k = pow(kernel::cuda::weight<kernel::cuda::PBFKERNEL>(h * 0.4f, inv_h), 4);
	cuda::pu::setConstantMemory(smoothing_length, particle_mass, inv_stable_density, k);

	for (int itr = 0; itr < num_iteration; ++itr) {
		cuda::calcScalingFactor(scaling_factor, interim_position, ns,
			inv_stable_density, particle_mass, smoothing_length, relaxation, num_particle);
		cuda::calcPositionUpdate(position_update, interim_position, scaling_factor, ns,
			inv_stable_density, particle_mass, smoothing_length, num_particle);
		cuda::responseCollision(position_update, interim_position, old_position, num_particle);
		cuda::updateInterimPosition(interim_position, position_update, num_particle);
	}
}

void solveConstraint(
	dom_dim* interim_position, dom_dim* position_update, scalar_t* scaling_factor,
	scalar_t* kernels, dom_dim* grad_kernels,
	const dom_dim* old_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_iteration,
	int num_particle
	)
{
	auto& h = smoothing_length;
	const auto inv_h = 1.f / h;
	static auto k = pow(kernel::cuda::weight<kernel::cuda::PBFKERNEL>(h * 0.4f, inv_h), 4);
	cuda::pu::setConstantMemory(smoothing_length, particle_mass, inv_stable_density, k);

	for (int itr = 0; itr < num_iteration; ++itr) {
		cuda::memorizeKernelCalc(kernels, grad_kernels, ns, interim_position, smoothing_length, num_particle);
		cuda::calcScalingFactor(scaling_factor, kernels, grad_kernels, ns,
			inv_stable_density, particle_mass, smoothing_length, relaxation, num_particle);
		cuda::calcPositionUpdate(position_update, scaling_factor, kernels, grad_kernels, ns,
			inv_stable_density, particle_mass, smoothing_length, num_particle);
		cuda::responseCollision(position_update, interim_position, old_position, num_particle);
		cuda::updateInterimPosition(interim_position, position_update, num_particle);
	}
}

void solveConstraint(
	dom_dim* interim_position, dom_dim* position_update, scalar_t* scaling_factor,
	scalar_t* kernels, dom_dim* grad_kernels,
	const dom_dim* old_position, const pbf_boundary& boundary,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_iteration,
	int num_particle
	)
{
	auto& h = smoothing_length;
	const auto inv_h = 1.f / h;
	static auto k = pow(kernel::cuda::weight<kernel::cuda::PBFKERNEL>(h * 0.4f, inv_h), 4);
	cuda::pu::setConstantMemory(smoothing_length, particle_mass, inv_stable_density, k);

	for (int itr = 0; itr < num_iteration; ++itr) {
		cuda::memorizeKernelCalc(kernels, grad_kernels, ns, interim_position, smoothing_length, num_particle);
		cuda::calcScalingFactor(scaling_factor, kernels, grad_kernels, ns,
			inv_stable_density, particle_mass, smoothing_length, relaxation, num_particle);
		cuda::calcPositionUpdate(position_update, scaling_factor, kernels, grad_kernels, ns,
			inv_stable_density, particle_mass, smoothing_length, num_particle);
		cuda::responseCollision(position_update, interim_position, old_position, num_particle,
			boundary.inner_spheres, boundary.inner_spheres_num, boundary.outer_spheres, boundary.outer_spheres_num,
			boundary.points_on_planes, boundary.normals_on_planes, boundary.planes_num);
		cuda::updateInterimPosition(interim_position, position_update, num_particle);
	}
}

void update(
	dom_dim* new_position, dom_dim* new_velocity,
	dom_dim* interim_velocity,
	const dom_dim* result_sorted_position, const dom_dim* old_sorted_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_time_step, scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t xsph_parameter,
	int num_particle
	)
{
	cuda::updateVelocity(interim_velocity, result_sorted_position, old_sorted_position, inv_time_step, num_particle);
	cuda::applyXSPH(new_velocity, result_sorted_position, interim_velocity, ns, smoothing_length,
		xsph_parameter, inv_stable_density, particle_mass, num_particle);
	//cuda::updateVelocity(new_velocity, interim_velocity, num_particle);
	cuda::updatePosition(new_position, result_sorted_position, num_particle);
}

void update(
	dom_dim* new_position, dom_dim* new_velocity,
	dom_dim* interim_velocity, dom_dim* interim_vorticity,
	const dom_dim* result_sorted_position, const dom_dim* old_sorted_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_time_step, scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t xsph_parameter, scalar_t vc_parameter,
	int num_particle
	)
{
	cuda::updateVelocity(interim_velocity, result_sorted_position, old_sorted_position, inv_time_step, num_particle);
	//cuda::calcVorticity(interim_vorticity, result_sorted_position, interim_velocity, ns, smoothing_length, particle_mass, inv_stable_density, num_particle);
	//cuda::addVorticityConfinement(interim_velocity, interim_vorticity, result_sorted_position, ns,
	//	smoothing_length, particle_mass, inv_stable_density, vc_parameter, 1.f / inv_time_step, num_particle);
	//cuda::applyXSPH(new_velocity, result_sorted_position, interim_velocity, ns, smoothing_length,
	//	xsph_parameter, inv_stable_density, particle_mass, num_particle);

	cuda::updateVelocity(new_velocity, interim_velocity, num_particle);	// no xsph

	cuda::updatePosition(new_position, result_sorted_position, num_particle);
}

void one_step(
	pbf_particle& simulatee,
	pbf_buffer& buf,
	const std::pair<dom_dim, dom_dim> simulation_area,
	int num_solver_iteration
	)
{
	// cuda state
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	// rename
	auto& phase = simulatee.phase;
	auto& param = simulatee.parameter;
	auto ns = simulatee.ns;
	uint32_t& num_particle = simulatee.phase.num;
	const auto inv_rho0 = 1.f / param.stable_density;
	const auto m = param.particle_mass;
	const auto h = param.smoothing_length;
	const auto inv_t = 1.f / param.time_step;

	predict(buf.interim.x, buf.interim.v, phase.x, phase.v, simulatee.external.body_force, param.time_step, num_particle);

	//ns.reorderDataAndFindCellStart(buf.sorted_predicted_pos, buf.interim.x, buf.sorted_old_pos, phase.x, num_particle);
	ns->detectNeighbors(buf.sorted_predicted_pos, buf.interim.x, buf.sorted_old_pos, phase.x, param.smoothing_length, simulation_area, num_particle);

	solveConstraint(buf.sorted_predicted_pos, buf.delta_position, buf.scaling_factor, 
		buf.kernels, buf.grad_kernels,
		buf.sorted_old_pos,
		ns, inv_rho0, m, h, param.relaxation, num_solver_iteration, num_particle);

	//update(phase.x, phase.v, buf.interim.v, buf.sorted_predicted_pos, buf.sorted_old_pos, ns, inv_t, inv_rho0, m, h, param.xsph_parameter, num_particle); // no vorticity confinement
	update(phase.x, phase.v, buf.interim.v, buf.vorticity, buf.sorted_predicted_pos, buf.sorted_old_pos, ns, inv_t, inv_rho0, m, h, param.xsph_parameter, param.vc_parameter, num_particle);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}

void one_step(
	pbf_particle& simulatee,
	pbf_buffer& buf,
	const pbf_boundary& boundary,
	const std::pair<dom_dim, dom_dim> simulation_area,
	int num_solver_iteration
	)
{
	// cuda state
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// rename
	auto& phase = simulatee.phase;
	auto& param = simulatee.parameter;
	auto ns = simulatee.ns;
	uint32_t& num_particle = simulatee.phase.num;
	const auto inv_rho0 = 1.f / param.stable_density;
	const auto m = param.particle_mass;
	const auto h = param.smoothing_length;
	const auto inv_t = 1.f / param.time_step;

	predict(buf.interim.x, buf.interim.v, phase.x, phase.v, simulatee.external.body_force, param.time_step, num_particle);

	//ns.reorderDataAndFindCellStart(buf.sorted_predicted_pos, buf.interim.x, buf.sorted_old_pos, phase.x, num_particle);
	ns->detectNeighbors(buf.sorted_predicted_pos, buf.interim.x, buf.sorted_old_pos, phase.x, param.smoothing_length, simulation_area, num_particle);

	solveConstraint(buf.sorted_predicted_pos, buf.delta_position, buf.scaling_factor,
		buf.kernels, buf.grad_kernels,
		buf.sorted_old_pos, boundary,
		ns, inv_rho0, m, h, param.relaxation, num_solver_iteration, num_particle);

	//update(phase.x, phase.v, buf.interim.v, buf.sorted_predicted_pos, buf.sorted_old_pos, ns, inv_t, inv_rho0, m, h, param.xsph_parameter, num_particle); // no vorticity confinement
	update(phase.x, phase.v, buf.interim.v, buf.vorticity, buf.sorted_predicted_pos, buf.sorted_old_pos, ns, inv_t, inv_rho0, m, h, param.xsph_parameter, param.vc_parameter, num_particle);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}

} // end of pbf ns
