#include "pbf_neighbor_search_device_util.cuh"

namespace pbf {
namespace cuda {

__host__ __device__ uint32_t getNeighborParticleAddr(
	const uint32_t* neighbor_list,
	uint32_t self_particle_index, uint32_t neighbor_particle_count, uint32_t max_pair)
{
	//const auto i = neighbor_particle_count * 32 + self_particle_index % 32 + (self_particle_index / 32) * 32 * max_pair;
	const auto i = (neighbor_particle_count << 5) + (self_particle_index & 31) + ((self_particle_index >> 5) << 5) * max_pair;
	return neighbor_list[i];
}

__host__ __device__ uint32_t getNeighborListIndex(
	uint32_t self_particle_index, uint32_t neighbor_particle_count, uint32_t max_pair)
{
	//const auto i = neighbor_particle_count * 32 + self_particle_index % 32 + (self_particle_index / 32) * 32 * max_pair;
	const auto i = (neighbor_particle_count << 5) + (self_particle_index & 31) + ((self_particle_index >> 5) << 5) * max_pair;
	return i;
}

}	// end of cuda ns
}	// end of pbf ns

