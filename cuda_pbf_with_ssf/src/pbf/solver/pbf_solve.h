#pragma once
#include "../pbf_type.h"
#include "../interaction/pbf_neighbor_search.h"

namespace pbf {

void predict(
	dom_dim* interim_position, dom_dim* interim_velocity,
	const dom_dim* position, const dom_dim* velocity,
	dom_dim ext_force, scalar_t time_step,
	int num_particle
	);

void solveConstraint(
	dom_dim* interim_position, dom_dim* position_update, scalar_t* scaling_factor,
	const dom_dim* old_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_iteration,
	int num_target_particle
	);

// kernel cache
void solveConstraint(
	dom_dim* interim_position, dom_dim* position_update, scalar_t* scaling_factor,
	scalar_t* kernels, dom_dim* grad_kernels,
	const dom_dim* old_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t relaxation,
	int num_iteration,
	int num_target_particle
	);

void update(
	dom_dim* new_position, dom_dim* new_velocity,
	dom_dim* interim_velocity,
	const dom_dim* result_sorted_position, const dom_dim* old_sorted_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_time_step, scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t xsph_parameter,
	int num_particle
	);

void update(
	dom_dim* new_position, dom_dim* new_velocity,
	dom_dim* interim_velocity, dom_dim* interim_vorticity,
	const dom_dim* result_sorted_position, const dom_dim* old_sorted_position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_time_step, scalar_t inv_stable_density, scalar_t particle_mass, scalar_t smoothing_length,
	scalar_t xsph_parameter, scalar_t vc_parameter,
	int num_particle
	);

void one_step(
	pbf_particle& simulatee,
	pbf_buffer& buffer,
	int num_solver_iteration
	);

} // end of pbf ns
