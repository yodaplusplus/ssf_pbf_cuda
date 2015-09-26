#pragma once
#include "../../pbf_type.h"
#include "../../interaction/pbf_neighbor_search.h"

namespace pbf {
namespace cuda {

void memorizeKernelCalc(
	scalar_t* kernels,
	dom_dim* grad_kernels,
	std::shared_ptr<neighbor_search>& ns,
	const dom_dim* position,
	scalar_t smoothing_length,
	int num_particle
	);

} // end of cuda ns
} // end of pbf ns
