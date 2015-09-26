#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda{

// calculate position in uniform grid
__host__ __device__
inline glm::ivec2 calcGridPos(glm::vec2 p, scalar_t cell_width)
{
	glm::ivec2 gridPos;
	gridPos.x = static_cast<int>(floor(p.x / cell_width));
	gridPos.y = static_cast<int>(floor(p.y / cell_width));

	return gridPos;
}

// calculate position in uniform grid
__host__ __device__
inline glm::ivec3 calcGridPos(const glm::vec3& p, scalar_t cell_width)
{
	glm::ivec3 gridPos;
	gridPos.x = static_cast<int>(floor(p.x / cell_width));
	gridPos.y = static_cast<int>(floor(p.y / cell_width));
	gridPos.z = static_cast<int>(floor(p.z / cell_width));

	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__host__ __device__
inline uint32_t calcGridHash(glm::ivec2 gridPos, glm::uvec2 grid_size)
{
	gridPos.x = gridPos.x & (grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (grid_size.y - 1);

	return gridPos.y * grid_size.x + gridPos.x;
}

// calculate address in grid from position (clamping to edges)
__host__ __device__
inline uint32_t calcGridHash(glm::ivec3 gridPos, glm::uvec3 grid_size)
{
	gridPos.x = gridPos.x & (grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (grid_size.y - 1);
	gridPos.z = gridPos.z & (grid_size.z - 1);

	return gridPos.z * grid_size.x * grid_size.y + gridPos.y * grid_size.x + gridPos.x;
}

// calculate address in grid from position (clamping to edges)
__host__ __device__
inline uint32_t calcGridHash(glm::ivec3 gridPos, uint3 grid_size)
{
	gridPos.x = gridPos.x & (grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (grid_size.y - 1);
	gridPos.z = gridPos.z & (grid_size.z - 1);

	return gridPos.z * grid_size.x * grid_size.y + gridPos.y * grid_size.x + gridPos.x;
}

}	// end of cuda ns
}	// end of pbf ns
