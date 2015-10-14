#pragma once
#include "../../../pbf_type.h"

namespace pbf {
namespace cuda {

__device__ void responseInnerSphereBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& center, float radius	// sphere definition
	);

__device__ void responseOuterSphereBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& center, float radius	// sphere definition
	);

}	// end of cuda ns
}	// end of pbf ns
