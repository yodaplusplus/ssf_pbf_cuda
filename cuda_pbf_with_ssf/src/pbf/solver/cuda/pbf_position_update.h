#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void calcPositionUpdate(
	dom_dim* position_update,
	const dom_dim* position,
	const scalar_t* scaling_factor,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	int num_target_particle
	);

// kernel cache
void calcPositionUpdate(
	dom_dim* position_update,
	const scalar_t* scaling_factor,
	const scalar_t* kernels,
	const dom_dim* grad_kernels,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	int num_target_particle
	);

namespace pu {
void setConstantMemory(
	scalar_t arg_h,
	scalar_t arg_m,
	scalar_t arg_inv_rho0,
	scalar_t arg_k
	);
} // end of pu ns

} // end of cuda ns
} // end of pbf ns
