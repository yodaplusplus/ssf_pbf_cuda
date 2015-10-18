#include "pbf_boundary.h"
#include "util/pbf_cuda_util.h"

void pbf_boundary::allocate(
	const std::vector<glm::vec4>& h_inner_spheres,	// float4(center, radius)
	const std::vector<glm::vec4>& h_outer_spheres,	// float4(center, radius)
	const std::vector<dom_dim>& h_points_on_planes,
	const std::vector<dom_dim>& h_normals_on_planes
	)
{
	// memory allocation
	inner_spheres_num = h_inner_spheres.size();
	cudaMalloc(&inner_spheres, sizeof(glm::vec4) * inner_spheres_num);
	outer_spheres_num = h_outer_spheres.size();
	cudaMalloc(&outer_spheres, sizeof(glm::vec4) * outer_spheres_num);
	planes_num = h_points_on_planes.size();
	cudaMalloc(&points_on_planes, sizeof(dom_dim) * planes_num);
	cudaMalloc(&normals_on_planes, sizeof(dom_dim) * planes_num);

	// copy
	cudaMemcpy(inner_spheres, h_inner_spheres.data(), sizeof(glm::vec4) * inner_spheres_num, cudaMemcpyHostToDevice);	
	cudaMemcpy(outer_spheres, h_outer_spheres.data(), sizeof(glm::vec4) * outer_spheres_num, cudaMemcpyHostToDevice);
	cudaMemcpy(points_on_planes, h_points_on_planes.data(), sizeof(dom_dim) * planes_num, cudaMemcpyHostToDevice);
	cudaMemcpy(normals_on_planes, h_normals_on_planes.data(), sizeof(dom_dim) * planes_num, cudaMemcpyHostToDevice);
}

void pbf_boundary::free()
{
	cudaFree(inner_spheres);
	cudaFree(outer_spheres);
	cudaFree(points_on_planes);
	cudaFree(normals_on_planes);
}
