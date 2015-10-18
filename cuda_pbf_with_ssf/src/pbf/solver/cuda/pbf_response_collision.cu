#include "pbf_response_collision.h"
#include "boundary/pbf_plane_boundary.cuh"
#include "boundary/pbf_sphere_boundary.cuh"
#include "../../util/pbf_cuda_util.h"
#include <device_launch_parameters.h>

namespace pbf {
namespace cuda {

namespace {

__global__ void responseCollisionCUDA(
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
	)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= num_particle) return;

	auto delta_p = position_update[index];
	auto predicted_x = predicted_position[index];
	auto old_x = predicted_position[index];

	while (true) {
		auto p = predicted_x + delta_p;
		dom_dim sim_origin(2.2f);
		dom_dim sim_end(6.5f);
		bool collision_check = false;

		// inner spheres
		for(uint32_t i = 0; i < inner_spheres_num; ++i) {
			auto sphere = inner_spheres[i];
			responseInnerSphereBoundary(collision_check, delta_p, old_x, p, dom_dim(sphere.x, sphere.y, sphere.z), sphere.w);
			p = predicted_x + delta_p;
		}

		// outer spheres
		for(uint32_t i = 0; i < outer_spheres_num; ++i) {
			auto sphere = outer_spheres[i];
			responseInnerSphereBoundary(collision_check, delta_p, old_x, p, dom_dim(sphere.x, sphere.y, sphere.z), sphere.w);
			p = predicted_x + delta_p;
		}

		// planes
		for(uint32_t i = 0; i < planes_num; ++i) {
			auto p_on_plane = points_on_planes[i];
			auto n_on_plane = normals_on_planes[i];
			responsePlaneBoundary(collision_check, delta_p, old_x, p, p_on_plane, n_on_plane);
			p = predicted_x + delta_p;
		}

		if (!collision_check) {
			break;
		}
	}
	position_update[index] = delta_p;
}

}	// end of unnamed ns

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
	)
{
	uint32_t num_thread, num_block;
	computeGridSize(num_particle, 128, num_block, num_thread);

	if (num_block > 0)
		responseCollisionCUDA<< < num_block, num_thread >> >
		(position_update, predicted_position, old_position, num_particle,
		inner_spheres, inner_spheres_num, outer_spheres, outer_spheres_num, points_on_planes, normals_on_planes, planes_num);

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

} // end of cuda ns
} // end of pbf ns


