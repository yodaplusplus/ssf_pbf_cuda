#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void applyExternalForce(
	dom_dim* interim_velocity,
	const dom_dim* velocity,
	dom_dim ext_force,
	scalar_t time_step,
	int num_particle
	);

void predictPosition(
	dom_dim* interim_position,
	const dom_dim* position,
	const dom_dim* velocity,
	scalar_t time_step,
	int num_particle);

} // end of cuda ns
} // end of pbf ns
