#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void calcDensity(
	const dom_dim* position,
	std::shared_ptr<neighbor_search>& ns,
	scalar_t particle_mass,
	scalar_t smoothing_length,
	int num_particle
	);

} // end of cuda ns
} // end of pbf ns

