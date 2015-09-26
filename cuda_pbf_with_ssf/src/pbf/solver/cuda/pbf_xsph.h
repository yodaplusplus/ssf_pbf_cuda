#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void applyXSPH(
	dom_dim* new_velocity,
	const dom_dim* sorted_position,
	const dom_dim* old_sorted_velocity,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t smoothing_length,
	scalar_t xsph_parameter,
	scalar_t inv_stable_density,
	scalar_t particle_mass,
	int num_target_particle
	);

} // end of cuda ns
} // end of pbf ns
