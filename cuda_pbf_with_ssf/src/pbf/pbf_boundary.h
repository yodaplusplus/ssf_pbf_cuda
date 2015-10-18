#pragma once
#include "pbf_type.h"

struct pbf_boundary {
	glm::vec4* inner_spheres;	// float4(center, radius)
	uint32_t inner_spheres_num;
	glm::vec4* outer_spheres;	// float4(center, radius)
	uint32_t outer_spheres_num;
	dom_dim* points_on_planes;
	dom_dim* normals_on_planes;
	uint32_t planes_num;
	void allocate(
		const std::vector<glm::vec4>& h_inner_spheres,	// float4(center, radius)
		const std::vector<glm::vec4>& h_outer_spheres,	// float4(center, radius)
		const std::vector<dom_dim>& h_points_on_planes,
		const std::vector<dom_dim>& h_normals_on_planes
		);
	void free();
};

