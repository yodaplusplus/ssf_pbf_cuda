#pragma once
#include "../../../pbf_type.h"

namespace pbf {
namespace cuda {

// if not intersected, no values are changed
__device__ void responsePlaneBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& on_plane, const glm::vec3& normal	// plane definition
	);

}	// end of cuda ns
}	// end of pbf ns

