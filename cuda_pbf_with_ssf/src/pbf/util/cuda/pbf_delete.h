#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void checkDeleteParticle(
	uint32_t* hash,
	const dom_dim* position,
	const std::pair<dom_dim, dom_dim>& domain,
	uint32_t delete_cell_id,
	uint32_t num_particle
	);

void calculateDeleteNumber(
	uint32_t* deleted_num,
	const uint32_t* hash,
	uint32_t delete_cell_id,
	uint32_t num_particle
	);

}	// end of cuda ns
}	// end of pbf ns

