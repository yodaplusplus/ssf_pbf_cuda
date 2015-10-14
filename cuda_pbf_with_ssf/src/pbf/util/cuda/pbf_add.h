#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void addParticle(
	pbf_particle& particle,
	const glm::vec3* adding_position,
	const glm::vec3* adding_velocity,
	uint32_t adding_num	// the number of elements, not byte
	);	

}	// end of cuda ns
}	// end of pbf ns

