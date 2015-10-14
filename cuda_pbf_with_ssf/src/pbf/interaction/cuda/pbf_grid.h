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

// calculate position in uniform grid
__host__ __device__
inline glm::ivec3 calcGridPos(const float4& p, scalar_t cell_width)
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

__host__ __device__
inline uint64_t splitBy3ZOrder(uint32_t a) {
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	// shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 32) & 0x1f00000000ffff;
	// shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 16) & 0x1f0000ff0000ff;
	// shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 8) & 0x100f00f00f00f00f; 
	// shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; 
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

// calculate address in grid from position
__host__ __device__
inline uint64_t calcGridHashZOrder(glm::ivec3 gridPos)
{
	uint64_t answer = 0;
	answer |= splitBy3ZOrder(gridPos.x) | splitBy3ZOrder(gridPos.y) << 1 | splitBy3ZOrder(gridPos.z) << 2;
	return answer;
}

}	// end of cuda ns
}	// end of pbf ns
