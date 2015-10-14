#include "pbf_plane_boundary.cuh"

namespace pbf {
namespace cuda {

__device__ void responsePlaneBoundary(bool& is_collided, glm::vec3& delta_p,
	const glm::vec3& old_p, const glm::vec3& predicted_p,	// particle position
	const glm::vec3& on_plane, const glm::vec3& normal	// plane definition
	)
{
	const auto pa = old_p - on_plane;
	const auto pb = predicted_p - on_plane;
	//const auto pa_n = glm::dot(pa, normal);
	const auto pa_n = fmaf(pa.z, normal.z, fmaf(pa.y, normal.y, pa.x * normal.x));
	const auto pb_n = fmaf(pb.z, normal.z, fmaf(pb.y, normal.y, pb.x * normal.x));
	glm::vec3 intersection;
	if (pa_n == 0.f && pb_n == 0.f) {
		// old_p and predicted_p both are on a plane
		return;
	}
	else if ((pa_n >= 0.f && pb_n < 0.f)) {	// particle intersects boundary
		is_collided = true;
		intersection = old_p + (predicted_p - old_p) * (abs(pa_n) / (abs(pa_n) + abs(pb_n)));
	}
	else if (pa_n < 0.f && pb_n < 0.f) {	// particle lies inside completely
		is_collided = true;
		const auto d = -fmaf(on_plane.z, normal.z, fmaf(on_plane.y, normal.y, on_plane.x * normal.x));
		const auto t = -(fmaf(predicted_p.z, normal.z, fmaf(predicted_p.y, normal.y, predicted_p.x * normal.x)) + d) /
			(fmaf(normal.z, normal.z, fmaf(normal.y, normal.y, normal.x * normal.x)));
		intersection = predicted_p + t * normal;
	}
	else {
		// no intersection, do nothing
		return;
	}

	const auto di = predicted_p - intersection;
	const auto constraint = fmaf(di.z, normal.z, fmaf(di.y, normal.y, di.x * normal.x));
	delta_p += -constraint * normal;

	//printf("old_p: %f, %f, %f\n", old_p.x, old_p.y, old_p.z);
	//printf("predicted_p: %f, %f, %f\n", predicted_p.x, predicted_p.y, predicted_p.z);
	//printf("intersection: %f, %f, %f\n", intersection.x, intersection.y, intersection.z);
	//printf("constraint: %f\n", constraint);
}

}	// end of cuda ns
}	// end of pbf ns
