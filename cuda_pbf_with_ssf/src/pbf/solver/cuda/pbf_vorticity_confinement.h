#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void calcVorticity(
	dom_dim* vorticity,
	const dom_dim* sorted_position,
	const dom_dim* sorted_velocity,
	neighbor_search& ns,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	scalar_t inv_stable_density,
	int num_particle
	);

void addVorticityConfinement(
	dom_dim* velocity,
	const dom_dim* vorticity,
	const dom_dim* sorted_position,
	neighbor_search& ns,
	scalar_t smoothing_length,
	scalar_t particle_mass,
	scalar_t inv_stable_density,
	scalar_t adjust_parameter,
	scalar_t time_step,
	int num_particle
	);

}	// end of cuda ns
}	// end of pbf ns
