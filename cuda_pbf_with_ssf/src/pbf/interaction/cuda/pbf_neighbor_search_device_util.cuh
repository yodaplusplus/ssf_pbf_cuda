#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

__host__ __device__ uint32_t getNeighborParticleAddr(
	const uint32_t* neighbor_list,
	uint32_t self_particle_index, uint32_t neighbor_particle_count, uint32_t max_pair_particle);

__host__ __device__ uint32_t getNeighborListIndex(
	uint32_t self_particle_index, uint32_t neighbor_particle_count, uint32_t max_pair_particle);

}	// end of cuda ns
}	// end of pbf ns
