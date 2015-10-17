#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void calcScalingFactor(
	scalar_t* scaling_factor,
	const dom_dim* position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	scalar_t relaxation,
	int num_particle
	);

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
	);

// embedded boundary
void responseCollision(
	dom_dim* position_update,
	const dom_dim* predicted_position,
	const dom_dim* old_position,
	int num_particle
	);

void updateInterimPosition(
	dom_dim* interim_position,
	const dom_dim* position_update,
	int num_particle
	);

} // end of cuda ns
} // end of pbf ns
