#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"
#include "../../solver/cuda/pbf_xsph.h"
#include "../../solver/cuda/pbf_vorticity_confinement.h"

namespace pbf {
namespace cuda {

void updateVelocity(
	dom_dim* velocity,
	const dom_dim* new_position,
	const dom_dim* old_position,
	scalar_t inv_time_step,
	int num_particle
	);

void updatePosition(
	dom_dim* new_position,
	const dom_dim* result_position,
	int num_particle
	);

// if xsph not used
void updateVelocity(
	dom_dim* new_vel,
	const dom_dim* result_vel,
	int num_particle
	);

} // end of cuda ns
} // end of pbf ns
