#include "pbf_sphere_boundary.cuh"
#include <device_launch_parameters.h>
#include <cmath>

namespace pbf {
namespace cuda {

__device__ void responseInnerSphereBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& center, float radius	// sphere definition
	)
{
	const auto center_diff = center - predicted_p;
	//const auto dist_off_center = glm::length(center_diff);
	const auto dist2_off_center = fmaf(center_diff.z, center_diff.z, fmaf(center_diff.y, center_diff.y, center_diff.x * center_diff.x));
	const auto reciprocal_dist_off_center = rsqrtf(dist2_off_center);
	if (dist2_off_center > radius * radius) {
		is_collided = true;
		delta_p.x = fmaf(center_diff.x, 1.0001f, delta_p.x);	// slightly pushing inward
		delta_p.y = fmaf(center_diff.y, 1.0001f, delta_p.y);
		delta_p.z = fmaf(center_diff.z, 1.0001f, delta_p.z);
		//const auto residual = fmaf(dist2_off_center, reciprocal_dist_off_center, - radius);
		const auto to_center = center_diff * reciprocal_dist_off_center;
		//delta_p += residual * to_center;
		delta_p.x = fmaf(-radius, to_center.x, delta_p.x);
		delta_p.y = fmaf(-radius, to_center.y, delta_p.y);
		delta_p.z = fmaf(-radius, to_center.z, delta_p.z);
		//delta_p.z = fmaf(residual, to_center.z, delta_p.z);
	}
	else {
		return;
	}

}

__device__ void responseOuterSphereBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& center, float radius	// sphere definition
	)
{
	const auto center_diff = center - predicted_p;
	//const auto dist_off_center = glm::length(center_diff);
	const auto dist2_off_center = fmaf(center_diff.z, center_diff.z, fmaf(center_diff.y, center_diff.y, center_diff.x * center_diff.x));
	const auto reciprocal_dist_off_center = rsqrtf(dist2_off_center);
	if (dist2_off_center < radius * radius) {
		is_collided = true;
		delta_p.x = fmaf(center_diff.x, 0.9999f, delta_p.x);	// slightly pushing outward
		delta_p.y = fmaf(center_diff.y, 0.9999f, delta_p.y);
		delta_p.z = fmaf(center_diff.z, 0.9999f, delta_p.z);
		//const auto residual = fmaf(-dist2_off_center, reciprocal_dist_off_center, radius);
		const auto to_center = center_diff * reciprocal_dist_off_center;
		//delta_p += residual * -to_center;
		//delta_p.x = fmaf(residual, -to_center.x, delta_p.x);
		//delta_p.y = fmaf(residual, -to_center.y, delta_p.y);
		//delta_p.z = fmaf(residual, -to_center.z, delta_p.z);
		delta_p.x = fmaf(-radius, to_center.x, delta_p.x);
		delta_p.y = fmaf(-radius, to_center.y, delta_p.y);
		delta_p.z = fmaf(-radius, to_center.z, delta_p.z);
	}
	else {
		return;
	}

}

}	// end of cuda ns
}	// end of pbf ns
