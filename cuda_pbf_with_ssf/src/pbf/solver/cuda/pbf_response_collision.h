#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void responseCollision(
	dom_dim* position_update,
	const dom_dim* predicted_position,
	const dom_dim* old_position,
	int num_particle,
	const glm::vec4* inner_spheres,	// float4(center, radius)
	uint32_t inner_spheres_num,
	const glm::vec4* outer_spheres,
	uint32_t outer_spheres_num,
	const dom_dim* points_on_planes,
	const dom_dim* normals_on_planes,
	uint32_t planes_num
	);

} // end of cuda ns
} // end of pbf ns
